package responses

import (
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

const hostedToolsRequest = `{"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],"tools":[{"type":"function","name":"exec_command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}}},{"type":"tool_search","execution":"client","description":"Tool discovery"},{"type":"web_search","external_web_access":true}]}`

func TestHostedToolsConvertToChatFunctions(t *testing.T) {
	out := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("test-model", []byte(hostedToolsRequest), true)
	root := gjson.ParseBytes(out)

	names := map[string]gjson.Result{}
	root.Get("tools").ForEach(func(_, tool gjson.Result) bool {
		names[tool.Get("function.name").String()] = tool
		return true
	})
	for _, name := range []string{"exec_command", "tool_search", "web_search"} {
		if _, ok := names[name]; !ok {
			t.Fatalf("missing chat tool %q in %s", name, out)
		}
	}
	if !strings.Contains(root.Get("tools").Raw, `"query"`) {
		t.Fatalf("hosted tool parameters missing query: %s", root.Get("tools").Raw)
	}
	if got := names["tool_search"].Get("function.description").String(); got != "Tool discovery" {
		t.Fatalf("tool_search description = %q", got)
	}
}

func TestToolSearchCallRestoresItemTypeAndObjectArguments(t *testing.T) {
	// Non-stream path: a chat tool_call for a tool_search declaration must come
	// back as tool_search_call with object arguments and execution, not as a
	// function_call with stringified arguments.
	chatResp := []byte(`{"id":"chatcmpl-1","created":1,"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"tool_search","arguments":"{\"query\":\"gitnexus\",\"limit\":3}"}}]}}]}`)
	out := ConvertOpenAIChatCompletionsResponseToOpenAIResponsesNonStream(t.Context(), "test-model", []byte(hostedToolsRequest), nil, chatResp, nil)
	item := gjson.GetBytes(out, "output.#(type==\"tool_search_call\")")
	if !item.Exists() {
		t.Fatalf("no tool_search_call item in %s", out)
	}
	if item.Get("arguments").IsObject() != true {
		t.Fatalf("tool_search_call.arguments not an object: %s", item.Raw)
	}
	if got := item.Get("arguments.query").String(); got != "gitnexus" {
		t.Fatalf("arguments.query = %q", got)
	}
	if got := item.Get("execution").String(); got != "client" {
		t.Fatalf("execution = %q", got)
	}
}

func TestWebSearchCallRestoresActionShape(t *testing.T) {
	chatResp := []byte(`{"id":"chatcmpl-2","created":1,"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"call_2","type":"function","function":{"name":"web_search","arguments":"{\"query\":\"codex cli\"}"}}]}}]}`)
	out := ConvertOpenAIChatCompletionsResponseToOpenAIResponsesNonStream(t.Context(), "test-model", []byte(hostedToolsRequest), nil, chatResp, nil)
	item := gjson.GetBytes(out, "output.#(type==\"web_search_call\")")
	if !item.Exists() {
		t.Fatalf("no web_search_call item in %s", out)
	}
	if got := item.Get("action.type").String(); got != "search" {
		t.Fatalf("action.type = %q", got)
	}
	if got := item.Get("action.query").String(); got != "codex cli" {
		t.Fatalf("action.query = %q", got)
	}
}

func TestHostedToolHistoryReplaysAsCallAndOutput(t *testing.T) {
	// A second-turn request replays the executed tool_search pair and a
	// web_search_call; the chat side must see matching assistant tool_calls
	// followed by tool outputs.
	req := []byte(`{"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]},
		{"type":"tool_search_call","id":"tsc_1","call_id":"call_ts","status":"completed","execution":"client","arguments":{"query":"fff"}},
		{"type":"tool_search_output","id":"tso_1","call_id":"call_ts","status":"completed","execution":"client","tools":[{"type":"namespace","name":"mcp__fff","tools":[{"type":"function","name":"find_files","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}]}]},
		{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"news"}},
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}`)
	out := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("test-model", req, true)
	root := gjson.ParseBytes(out)

	var toolCalls []gjson.Result
	var toolMsgs []gjson.Result
	root.Get("messages").ForEach(func(_, m gjson.Result) bool {
		if m.Get("role").String() == "assistant" {
			m.Get("tool_calls").ForEach(func(_, tc gjson.Result) bool {
				toolCalls = append(toolCalls, tc)
				return true
			})
		}
		if m.Get("role").String() == "tool" {
			toolMsgs = append(toolMsgs, m)
		}
		return true
	})
	if len(toolCalls) != 2 {
		t.Fatalf("expected 2 replayed tool calls, got %d: %s", len(toolCalls), root.Get("messages").Raw)
	}
	if len(toolMsgs) != 2 {
		t.Fatalf("expected 2 tool outputs (incl. synthesized web_search ack), got %d: %s", len(toolMsgs), root.Get("messages").Raw)
	}
	if got := toolCalls[0].Get("function.name").String(); got != "tool_search" {
		t.Fatalf("first call name = %q", got)
	}
	if !strings.Contains(toolCalls[0].Get("function.arguments").String(), "fff") {
		t.Fatalf("tool_search arguments lost: %s", toolCalls[0].Raw)
	}
	if !strings.Contains(toolMsgs[0].Get("content").String(), "mcp__fff") {
		t.Fatalf("tool_search_output tools lost: %s", toolMsgs[0].Raw)
	}
	// Discovered tools must also enter the declared chat tool list, or the
	// upstream model cannot emit calls for them.
	discovered := false
	root.Get("tools").ForEach(func(_, tool gjson.Result) bool {
		if tool.Get("function.name").String() == "mcp__fff__find_files" {
			discovered = true
		}
		return true
	})
	if !discovered {
		t.Fatalf("discovered tool mcp__fff__find_files missing from chat tools: %s", root.Get("tools").Raw)
	}
}
