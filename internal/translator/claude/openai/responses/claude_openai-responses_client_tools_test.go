package responses

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

// codexClientTools are the apply_patch and tool_search declarations codex-cli
// 0.157.1 sends (descriptions shortened, shapes as captured).
const codexClientTools = `[
	{"type":"function","name":"exec_command","description":"Runs a command.","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}},
	{"type":"custom","name":"apply_patch","description":"Use apply_patch to edit files.","format":{"type":"grammar","syntax":"lark","definition":"start: begin_patch hunk+ end_patch"}},
	{"type":"tool_search","execution":"client","description":"# Tool discovery","parameters":{"type":"object","properties":{"limit":{"type":"number"},"query":{"type":"string"}},"required":["query"],"additionalProperties":false}}
]`

const codexUserTurn = `{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}`

// codexToolSearchFollowUp is the next Codex turn after a client tool_search:
// loaded MCP tools arrive only inside tool_search_output.tools.
var codexToolSearchFollowUp = []byte(`{"model":"claude-test","tools":` + codexClientTools + `,"input":[` + codexUserTurn + `,
	{"type":"tool_search_call","call_id":"toolu_search1","status":"completed","execution":"client","arguments":{"query":"gitnexus impact","limit":3}},
	{"type":"tool_search_output","call_id":"toolu_search1","status":"completed","execution":"client","tools":[
		{"type":"namespace","name":"mcp__gitnexus","description":"GitNexus","tools":[
			{"type":"function","name":"impact","description":"Blast radius","strict":false,"defer_loading":true,"parameters":{"type":"object","properties":{"target":{"type":"string"}},"required":["target"],"additionalProperties":false}}
		]}
	]}
]}`)

var codexClientToolsRequest = []byte(`{"model":"claude-test","tools":` + codexClientTools + `,"input":[` + codexUserTurn + `]}`)

// claudeToolUse is one streamed Claude tool_use block whose JSON input arrives
// in the given input_json_delta fragments.
type claudeToolUse struct {
	id, name  string
	fragments []string
}

func claudeToolUseStream(uses ...claudeToolUse) []string {
	chunks := []string{`data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":1,"output_tokens":0}}}`}
	for i, use := range uses {
		index := string(rune('0' + i))
		chunks = append(chunks, `data: {"type":"content_block_start","index":`+index+`,"content_block":{"type":"tool_use","id":"`+use.id+`","name":"`+use.name+`","input":{}}}`)
		for _, fragment := range use.fragments {
			chunks = append(chunks, `data: {"type":"content_block_delta","index":`+index+`,"delta":{"type":"input_json_delta","partial_json":`+mustJSONString(fragment)+`}}`)
		}
		chunks = append(chunks, `data: {"type":"content_block_stop","index":`+index+`}`)
	}
	return append(chunks,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`,
		`data: {"type":"message_stop"}`)
}

func mustJSONString(s string) string {
	out, _ := json.Marshal(s)
	return string(out)
}

// streamedToolCalls collects the per-call stream events, the response.completed
// output and the non-stream output for one Claude response.
type streamedToolCalls struct {
	added, done          []gjson.Result
	argumentEvents       []string
	completed, nonStream []gjson.Result
}

func toolCallOutputs(t *testing.T, request []byte, chunks []string) streamedToolCalls {
	t.Helper()
	var calls streamedToolCalls
	var param any
	for _, chunk := range chunks {
		for _, output := range ConvertClaudeResponseToOpenAIResponses(context.Background(), "claude-test", request, nil, []byte(chunk), &param) {
			event, data := parseClaudeResponsesSSEEvent(t, output)
			switch event {
			case "response.output_item.added":
				calls.added = append(calls.added, data.Get("item"))
			case "response.output_item.done":
				calls.done = append(calls.done, data.Get("item"))
			case "response.function_call_arguments.delta", "response.function_call_arguments.done":
				calls.argumentEvents = append(calls.argumentEvents, data.Get("item_id").String())
			case "response.completed":
				calls.completed = data.Get("response.output").Array()
			}
		}
	}
	out := ConvertClaudeResponseToOpenAIResponsesNonStream(context.Background(), "claude-test", request, nil, []byte(strings.Join(chunks, "\n")), nil)
	calls.nonStream = gjson.GetBytes(out, "output").Array()
	return calls
}

// finalItems returns call i as seen in output_item.done, response.completed and
// the non-stream response.
func (c streamedToolCalls) finalItems(t *testing.T, marker string, i int) map[string]gjson.Result {
	t.Helper()
	if len(c.done) <= i || len(c.completed) <= i || len(c.nonStream) <= i {
		t.Fatalf("%s: call %d missing: done=%d completed=%d non-stream=%d", marker, i, len(c.done), len(c.completed), len(c.nonStream))
	}
	return map[string]gjson.Result{"done": c.done[i], "completed": c.completed[i], "non-stream": c.nonStream[i]}
}

func TestCodexApplyPatchToolReachesClaude(t *testing.T) {
	out := ConvertOpenAIResponsesRequestToClaude("claude-test", codexClientToolsRequest, false)
	tool := gjson.GetBytes(out, `tools.#(name=="apply_patch")`)
	if !tool.Exists() || tool.Get("input_schema.required.0").String() != "input" || tool.Get("input_schema.properties.input.type").String() != "string" {
		t.Fatalf("APPLY_PATCH_MISSING: tools = %s", gjson.GetBytes(out, "tools").Raw)
	}
	if !strings.Contains(tool.Get("description").String(), "start: begin_patch hunk+ end_patch") {
		t.Fatalf("APPLY_PATCH_MISSING: grammar absent from description %q", tool.Get("description").String())
	}
}

func TestCodexApplyPatchCallReturnsCustomToolCall(t *testing.T) {
	patch := "*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch\n"
	calls := toolCallOutputs(t, codexClientToolsRequest, claudeToolUseStream(claudeToolUse{"toolu_ap", "apply_patch", []string{`{"input":` + mustJSONString(patch) + `}`}}))
	for label, item := range calls.finalItems(t, "APPLY_PATCH_NOT_CUSTOM_CALL", 0) {
		if item.Get("type").String() != "custom_tool_call" || item.Get("name").String() != "apply_patch" || item.Get("input").String() != patch {
			t.Fatalf("APPLY_PATCH_NOT_CUSTOM_CALL: %s item = %s", label, item.Raw)
		}
	}
}

func TestCodexToolSearchToolReachesClaude(t *testing.T) {
	out := ConvertOpenAIResponsesRequestToClaude("claude-test", codexClientToolsRequest, false)
	tool := gjson.GetBytes(out, `tools.#(name=="tool_search")`)
	if !tool.Exists() || tool.Get("input_schema.required.0").String() != "query" || tool.Get("description").String() != "# Tool discovery" {
		t.Fatalf("TOOL_SEARCH_MISSING: tools = %s", gjson.GetBytes(out, "tools").Raw)
	}
}

func TestCodexToolSearchCallReturnsToolSearchCall(t *testing.T) {
	patch := "*** Begin Patch\n*** Delete File: a.txt\n*** End Patch\n"
	calls := toolCallOutputs(t, codexClientToolsRequest, claudeToolUseStream(
		claudeToolUse{"toolu_ts", "tool_search", []string{`{"query":"gitnexus `, `impact","limit":3}`}},
		claudeToolUse{"toolu_ap", "apply_patch", []string{`{"input":` + mustJSONString(patch) + `}`}},
	))
	if len(calls.added) != 2 || calls.added[0].Get("type").String() != "tool_search_call" || calls.added[0].Get("call_id").String() != "toolu_ts" ||
		calls.added[0].Get("execution").String() != "client" {
		t.Fatalf("TOOL_SEARCH_NOT_CALL: added = %v", calls.added)
	}
	if len(calls.argumentEvents) != 0 {
		t.Fatalf("TOOL_SEARCH_NOT_CALL: function_call_arguments events for %v", calls.argumentEvents)
	}
	for label, item := range calls.finalItems(t, "TOOL_SEARCH_NOT_CALL", 0) {
		if item.Get("type").String() != "tool_search_call" || item.Get("call_id").String() != "toolu_ts" ||
			item.Get("execution").String() != "client" || !item.Get("arguments").IsObject() ||
			item.Get("arguments.query").String() != "gitnexus impact" || item.Get("arguments.limit").Int() != 3 {
			t.Fatalf("TOOL_SEARCH_NOT_CALL: %s item = %s", label, item.Raw)
		}
	}
	for label, item := range calls.finalItems(t, "TOOL_SEARCH_NOT_CALL", 1) {
		if item.Get("type").String() != "custom_tool_call" || item.Get("call_id").String() != "toolu_ap" || item.Get("input").String() != patch {
			t.Fatalf("TOOL_SEARCH_NOT_CALL: second call %s item = %s", label, item.Raw)
		}
	}
}

func TestToolSearchCallFollowsWinningDeclaration(t *testing.T) {
	const searchTool = `{"type":"tool_search","execution":"client","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}`
	const functionTool = `{"type":"function","name":"tool_search","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}`
	loadedFunction := `,{"type":"tool_search_output","call_id":"toolu_prev","status":"completed","execution":"client","tools":[` + functionTool + `]}`
	for _, tc := range []struct {
		label, tools, history, want string
	}{
		{"function only", functionTool, "", "function_call"},
		{"tool_search first", searchTool + "," + functionTool, "", "tool_search_call"},
		{"function first", functionTool + "," + searchTool, "", "function_call"},
		{"loaded function loses to top-level tool_search", searchTool, loadedFunction, "tool_search_call"},
	} {
		request := []byte(`{"model":"claude-test","tools":[` + tc.tools + `],"input":[` + codexUserTurn + tc.history + `]}`)
		calls := toolCallOutputs(t, request, claudeToolUseStream(claudeToolUse{"toolu_x", "tool_search", []string{`{"query":"x"}`}}))
		for label, item := range calls.finalItems(t, "TOOL_SEARCH_NOT_CALL", 0) {
			if item.Get("type").String() != tc.want {
				t.Fatalf("TOOL_SEARCH_NOT_CALL: %s %s item = %s", tc.label, label, item.Raw)
			}
			if tc.want == "function_call" && item.Get("arguments").String() != `{"query":"x"}` {
				t.Fatalf("TOOL_SEARCH_NOT_CALL: %s %s arguments = %s", tc.label, label, item.Raw)
			}
		}
	}
}

func TestCodexToolSearchHistoryReachesClaude(t *testing.T) {
	out := ConvertOpenAIResponsesRequestToClaude("claude-test", codexToolSearchFollowUp, false)
	messages := gjson.GetBytes(out, "messages").Array()
	if len(messages) != 3 {
		t.Fatalf("TOOL_SEARCH_HISTORY_DROPPED: messages = %s", gjson.GetBytes(out, "messages").Raw)
	}
	use := messages[1].Get("content.0")
	if messages[1].Get("role").String() != "assistant" || use.Get("type").String() != "tool_use" || use.Get("name").String() != "tool_search" ||
		use.Get("id").String() != "toolu_search1" || use.Get("input.query").String() != "gitnexus impact" || use.Get("input.limit").Int() != 3 {
		t.Fatalf("TOOL_SEARCH_HISTORY_DROPPED: tool_use = %s", messages[1].Raw)
	}
	result := messages[2].Get("content.0")
	if messages[2].Get("role").String() != "user" || result.Get("type").String() != "tool_result" || result.Get("tool_use_id").String() != "toolu_search1" ||
		!strings.Contains(result.Raw, "mcp__gitnexus__impact") {
		t.Fatalf("TOOL_SEARCH_HISTORY_DROPPED: tool_result = %s", messages[2].Raw)
	}
	if tool := gjson.GetBytes(out, `tools.#(name=="mcp__gitnexus__impact")`); tool.Get("input_schema.required.0").String() != "target" {
		t.Fatalf("TOOL_SEARCH_HISTORY_DROPPED: loaded tool missing from tools = %s", gjson.GetBytes(out, "tools").Raw)
	}
}

func TestCodexDeferredToolCallKeepsNamespace(t *testing.T) {
	calls := toolCallOutputs(t, codexToolSearchFollowUp, claudeToolUseStream(claudeToolUse{"toolu_impact", "mcp__gitnexus__impact", []string{`{"target":"Foo"}`}}))
	for label, item := range calls.finalItems(t, "DEFERRED_CALL_NAMESPACE_LOST", 0) {
		if item.Get("type").String() != "function_call" || item.Get("name").String() != "impact" || item.Get("namespace").String() != "mcp__gitnexus" {
			t.Fatalf("DEFERRED_CALL_NAMESPACE_LOST: %s item = %s", label, item.Raw)
		}
	}
}

func TestCodexSuccessiveToolSearchesStayUsable(t *testing.T) {
	loaded := func(callID, child string) string {
		return `{"type":"tool_search_call","call_id":"` + callID + `","status":"completed","execution":"client","arguments":{"query":"` + child + `"}},
			{"type":"tool_search_output","call_id":"` + callID + `","status":"completed","execution":"client","tools":[{"type":"namespace","name":"mcp__gitnexus","description":"GitNexus","tools":[
				{"type":"function","name":"` + child + `","defer_loading":true,"parameters":{"type":"object","properties":{"target":{"type":"string"}}}}]}]}`
	}
	request := []byte(`{"model":"claude-test","tools":` + codexClientTools + `,"input":[` + codexUserTurn + `,` +
		loaded("toolu_s1", "impact") + `,` + loaded("toolu_s2", "context") + `,` + loaded("toolu_s3", "impact") + `]}`)
	out := ConvertOpenAIResponsesRequestToClaude("claude-test", request, false)

	var searchIDs []string
	gjson.GetBytes(out, "messages").ForEach(func(_, message gjson.Result) bool {
		message.Get("content").ForEach(func(_, block gjson.Result) bool {
			if block.Get("type").String() == "tool_use" && block.Get("name").String() == "tool_search" {
				searchIDs = append(searchIDs, block.Get("id").String())
			}
			return true
		})
		return true
	})
	if strings.Join(searchIDs, ",") != "toolu_s1,toolu_s2,toolu_s3" {
		t.Fatalf("SUCCESSIVE_SEARCH_LOST: replayed searches = %v in %s", searchIDs, gjson.GetBytes(out, "messages").Raw)
	}
	if got := gjson.GetBytes(out, `tools.#(name=="mcp__gitnexus__impact")#`).Array(); len(got) != 1 {
		t.Fatalf("SUCCESSIVE_SEARCH_LOST: impact declared %d times in %s", len(got), gjson.GetBytes(out, "tools").Raw)
	}
	if !gjson.GetBytes(out, `tools.#(name=="mcp__gitnexus__context")`).Exists() {
		t.Fatalf("SUCCESSIVE_SEARCH_LOST: later loaded tool missing from %s", gjson.GetBytes(out, "tools").Raw)
	}
	calls := toolCallOutputs(t, request, claudeToolUseStream(claudeToolUse{"toolu_impact", "mcp__gitnexus__impact", []string{`{"target":"Foo"}`}}))
	for label, item := range calls.finalItems(t, "SUCCESSIVE_SEARCH_LOST", 0) {
		if item.Get("type").String() != "function_call" || item.Get("name").String() != "impact" || item.Get("namespace").String() != "mcp__gitnexus" {
			t.Fatalf("SUCCESSIVE_SEARCH_LOST: %s item = %s", label, item.Raw)
		}
	}
}

// Only client-executed tool_search is translated; server execution stays
// untranslated as before.
func TestServerToolSearchRemainsUntranslated(t *testing.T) {
	for _, execution := range []string{"", `,"execution":"server"`} {
		request := []byte(`{"model":"claude-test","tools":[{"type":"tool_search"` + execution + `}],"input":[` + codexUserTurn + `,
			{"type":"tool_search_call","id":"tsc_server","call_id":null,"arguments":{"query":"x"}` + execution + `},
			{"type":"tool_search_output","id":"tso_server","call_id":null,"tools":[{"type":"function","name":"server_loaded","parameters":{"type":"object"}}]` + execution + `}
		]}`)
		out := ConvertOpenAIResponsesRequestToClaude("claude-test", request, false)
		for _, name := range []string{"tool_search", "server_loaded"} {
			if gjson.GetBytes(out, `tools.#(name=="`+name+`")`).Exists() {
				t.Fatalf("SERVER_TOOL_SEARCH_TRANSLATED: execution %q exposed tool %q", execution, name)
			}
		}
		if messages := gjson.GetBytes(out, "messages").Raw; messages != `[{"role":"user","content":"hi"}]` {
			t.Fatalf("SERVER_TOOL_SEARCH_TRANSLATED: execution %q history became %s", execution, messages)
		}
	}
}
