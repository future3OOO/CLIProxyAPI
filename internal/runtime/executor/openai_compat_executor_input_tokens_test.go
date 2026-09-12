package executor

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
)

// TestOpenAICompatExecutorClaudeInputTokensEstimatesMissingUsage drives the
// real ExecuteStream seam: when the upstream never reports input usage, the
// pre-request estimate must still land on a client-bound usage event.
func TestOpenAICompatExecutorClaudeInputTokensEstimatesMissingUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"}}]}\n\n"+
			"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"+
			"data: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": server.URL + "/v1",
		"api_key":  "test",
	}}
	var logBuf strings.Builder
	origLog := log.StandardLogger().Out
	log.SetOutput(&logBuf)
	defer log.SetOutput(origLog)

	claudeRequest := []byte(`{"model":"compatible-model","stream":true,"messages":[{"role":"user","content":"Hello."}]}`)
	result, err := executor.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "compatible-model",
		Payload: claudeRequest,
	}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FormatClaude,
		Stream:          true,
		OriginalRequest: claudeRequest,
	})
	if err != nil {
		t.Fatalf("ExecuteStream error: %v", err)
	}
	var streamed strings.Builder
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
		streamed.Write(chunk.Payload)
	}

	body := streamed.String()
	var sawMessageStart, sawEstimated bool
	for _, line := range strings.Split(body, "\n") {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
		switch gjson.Get(payload, "type").String() {
		case "message_start":
			sawMessageStart = true
		}
		if tokens := gjson.Get(payload, "message.usage.input_tokens"); tokens.Exists() && tokens.Int() > 0 {
			sawEstimated = true
		}
		if tokens := gjson.Get(payload, "usage.input_tokens"); tokens.Exists() && tokens.Int() > 0 {
			sawEstimated = true
		}
	}
	if !sawMessageStart {
		t.Fatalf("no message_start in output: %q", body)
	}
	if !sawEstimated {
		t.Fatalf("no usage event carried estimated input_tokens: %q\nlogs: %s", body, logBuf.String())
	}
}

// TestOpenAICompatExecutorClaudeInputTokensDoneFirstStream drives the real
// ExecuteStream seam on a degenerate stream whose first translated event is a
// terminal usage-bearing message_delta (upstream [DONE] with no content). The
// terminal wait on the in-flight estimate is the intended contract: the
// correction lands on the only usage event the client ever sees.
func TestOpenAICompatExecutorClaudeInputTokensDoneFirstStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": server.URL + "/v1",
		"api_key":  "test",
	}}

	claudeRequest := []byte(`{"model":"compatible-model","stream":true,"messages":[{"role":"user","content":"Hello."}]}`)
	result, err := executor.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "compatible-model",
		Payload: claudeRequest,
	}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FormatClaude,
		Stream:          true,
		OriginalRequest: claudeRequest,
	})
	if err != nil {
		t.Fatalf("ExecuteStream error: %v", err)
	}
	var streamed strings.Builder
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
		streamed.Write(chunk.Payload)
	}

	body := streamed.String()
	var sawDelta bool
	for _, line := range strings.Split(body, "\n") {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
		if gjson.Get(payload, "type").String() != "message_delta" {
			continue
		}
		sawDelta = true
		if tokens := gjson.Get(payload, "usage.input_tokens"); !tokens.Exists() || tokens.Int() <= 0 {
			t.Fatalf("DONE-first message_delta lost the estimate: usage.input_tokens = %v\n%q", tokens.Value(), body)
		}
	}
	if !sawDelta {
		t.Fatalf("no message_delta in DONE-first output: %q", body)
	}
}

// TestOpenAICompatExecutorClaudeInputTokensKeepsCacheReadDelta drives the real
// ExecuteStream seam on a fully cache-read turn: upstream reports
// input_tokens=0 with cache_read_input_tokens>0, which is accounted usage —
// the zero is real ("zero new non-cached tokens"), not unknown. The pending
// estimate must not overwrite it, or input_tokens + cache_read_input_tokens
// double-counts the cached prefix.
//
// The request is deliberately large so the real tokenizer estimate stays
// pending through the localhost upstream round trip. The unpatched
// message_start assertion proves the pending path was taken — if the estimate
// ever resolved before message_start, that assertion fails instead of letting
// the test pass without exercising the guard.
func TestOpenAICompatExecutorClaudeInputTokensKeepsCacheReadDelta(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"}}]}\n\n"+
			"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":24,\"total_tokens\":24,\"prompt_tokens_details\":{\"cached_tokens\":6476}}}\n\n"+
			"data: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": server.URL + "/v1",
		"api_key":  "test",
	}}

	largeContent := strings.Repeat("the quick brown fox jumps over the lazy dog. ", 40000)
	claudeRequest := []byte(fmt.Sprintf(
		`{"model":"compatible-model","stream":true,"messages":[{"role":"user","content":%q}]}`,
		largeContent,
	))
	result, err := executor.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "compatible-model",
		Payload: claudeRequest,
	}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FormatClaude,
		Stream:          true,
		OriginalRequest: claudeRequest,
	})
	if err != nil {
		t.Fatalf("ExecuteStream error: %v", err)
	}
	var streamed strings.Builder
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
		streamed.Write(chunk.Payload)
	}

	body := streamed.String()
	var sawStart, sawDelta bool
	for _, line := range strings.Split(body, "\n") {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
		switch gjson.Get(payload, "type").String() {
		case "message_start":
			sawStart = true
			if tokens := gjson.Get(payload, "message.usage.input_tokens"); tokens.Int() != 0 {
				t.Fatalf("estimate resolved before message_start; pending path not exercised: %v\n%q", tokens.Value(), body)
			}
		case "message_delta":
			sawDelta = true
			if cached := gjson.Get(payload, "usage.cache_read_input_tokens"); cached.Int() != 6476 {
				t.Fatalf("cache_read_input_tokens lost: got %v\n%q", cached.Value(), body)
			}
			if tokens := gjson.Get(payload, "usage.input_tokens"); tokens.Int() != 0 {
				t.Fatalf("delta input_tokens equals the estimate instead of 0: got %v\n%q", tokens.Value(), body)
			}
		}
	}
	if !sawStart {
		t.Fatalf("no message_start in output: %q", body)
	}
	if !sawDelta {
		t.Fatalf("no message_delta in output: %q", body)
	}
}
