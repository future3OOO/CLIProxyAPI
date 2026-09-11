package claude

import (
	"testing"

	"github.com/tidwall/gjson"
)

func TestUserIDForwarded(t *testing.T) {
	in := []byte(`{"model":"swe-2-max","max_tokens":10,"metadata":{"user_id":"{\"session_id\":\"abc-123\"}"},"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	out := ConvertClaudeRequestToOpenAI("swe-2-max", in, false)
	if got := gjson.GetBytes(out, "user").String(); got == "" {
		t.Fatalf("user not forwarded: %s", out)
	}
}
