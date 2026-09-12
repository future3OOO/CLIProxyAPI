package claude

import (
	"testing"

	"github.com/tidwall/gjson"
)

func TestUserIDForwarded(t *testing.T) {
	in := []byte(`{"model":"swe-2-max","max_tokens":10,"metadata":{"user_id":"{\"session_id\":\"abc-123\"}"},"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	out := ConvertClaudeRequestToOpenAI("swe-2-max", in, false)
	if got := gjson.GetBytes(out, "user").String(); got != `{"session_id":"abc-123"}` {
		t.Fatalf("USER_VALUE_MISMATCH: user = %q, want exact session id: %s", got, out)
	}
}

func TestTopLevelUserForwardedWithoutMetadata(t *testing.T) {
	in := []byte(`{"model":"swe-2-max","max_tokens":10,"user":"billing-user-9","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	out := ConvertClaudeRequestToOpenAI("swe-2-max", in, false)
	if got := gjson.GetBytes(out, "user").String(); got != "billing-user-9" {
		t.Fatalf("top-level user not forwarded: user = %q: %s", got, out)
	}
}

func TestUserIDPrecedence(t *testing.T) {
	in := []byte(`{"model":"swe-2-max","max_tokens":10,"user":"billing-user-9","metadata":{"user_id":"{\"session_id\":\"abc-123\"}"},"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	out := ConvertClaudeRequestToOpenAI("swe-2-max", in, false)
	if got := gjson.GetBytes(out, "user").String(); got != `{"session_id":"abc-123"}` {
		t.Fatalf("USER_PRECEDENCE_LOST: user = %q, want session id to win: %s", got, out)
	}
}
