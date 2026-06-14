# Tier-2 tool-calling fix: multi-step / multi-hop on the OpenAI-compat endpoint

## Summary

This is a Tier-2 robust fix for multi-step / multi-hop tool calling on the
OpenAI-compatible endpoint (`POST /v1/chat/completions`, handled in
`api_endpoints/lm_studio/chat.py`). Straico exposes no native tool-calling API,
so tools are emulated by prompt injection: the model is taught to emit a
` ```json {"tool_calls":[...]}``` ` block, and the proxy parses that block back
into an OpenAI `tool_calls` envelope. Three root causes that broke multi-hop
(conversation flattening, tools being nulled on tool-result turns, and a brittle
single-block extractor) were fixed by porting the repair logic server-side into
`api_endpoints/response_utils/__init__.py` and rewiring the chat handler. The fix
greatly improves reliability but cannot make emulated tool calling as guaranteed
as a native Anthropic/OpenAI tool endpoint; the residual ceiling is documented
below.

## The three root causes fixed

### (a) Conversation flattening → role-tagged transcript

The old handler passed `json.dumps(msg, indent=True, ensure_ascii=False)` as the
single Straico `message` for every branch — the model received the entire OpenAI
`messages` array as one JSON blob with no role structure, which a base model
reads as an *example payload* rather than as the conversation it must continue.

Fix: `render_chat_transcript(messages)` in
`api_endpoints/response_utils/__init__.py` maps OpenAI roles to a clearly
role-tagged transcript string. In `api_endpoints/lm_studio/chat.py` the prompt
text is now computed as
`render_chat_transcript(msg)` for the tools branch and the plain (`else`) branch,
while the structured-output branch deliberately keeps `json.dumps(msg, ...)`
(see Role-mapping approach below). This replaces the three former `json.dumps(msg)`
completion sites.

### (b) Tools nulled on tool-result turns → channel kept open + multi-step cue

The old handler detected a trailing `tool` role and set `tools=None`, appending a
"Please interpret the answer" system message. With `tools` nulled, the
tool-parse gate never fired again, so the model could not be asked for the *next*
tool call — multi-hop died after the first hop.

Fix: that nulling block was removed, so the tool channel stays open on
tool-result turns and the tool-parse gate (`if tools is not None and len(tools)
!= 0:` in `api_endpoints/lm_studio/chat.py`) fires on every turn. The
tool-instruction system prompt gained an explicit multi-step continuation line:

> If previous tool results are present in the conversation, either call the NEXT
> required tool in the same JSON format, or—if you already have enough
> information—reply with the final answer to the user as plain text (NOT JSON).

On a final-answer turn the model replies in plain text, the extractor returns
`None`, and the request falls through to a normal completion.

### (c) Brittle single-block extractor → robust `extract_tool_calls`

The old inline extractor split on the first `{"tool_calls":[` and did not strip a
trailing ` ``` ` fence; a separate fallback fabricated a tool call from *any*
non-`tool_calls` dict by wrapping it under `tools[0]["function"]["name"]`.

Fix: `extract_tool_calls(text, keep_first_across_blocks=None)` in
`api_endpoints/response_utils/__init__.py` is a pure function returning a list of
`{"type":"function","function":{"name","arguments"}}` dicts, or `None` when no
tool call is detected — it never fabricates a call. The chat handler uses its
result to build the `tool_calls` envelope (random ids, `finish_reason:
"tool_calls"`, stream and non-stream paths).

## Role-mapping approach

**Why a transcript string and not structured messages.** Straico's
prompt-completion API accepts only a single `message` string. In the vendored
client (`aio_straico/api/v0.py`) the request body is literally
`json_body = {"message": message}` — there is no `system`/`instruction`
parameter and no messages-array method on `prompt_completion`. A faithful
role-mapping therefore has to render the conversation *into that one string*; it
cannot hand Straico a structured array.

**Transcript format** (produced by `render_chat_transcript`):

- A one-line neutral framing preamble telling the model this is the real
  conversation so far (its own earlier turns included), not an example, and that
  it should continue ONLY the final assistant turn.
- A leading `===== SYSTEM =====` section gathering all `system` messages.
- Role-tagged turns: `===== USER =====`, `===== ASSISTANT =====` (assistant
  `tool_calls` are rendered back as a ` ```json {"tool_calls":[...]}``` ` block),
  and `===== TOOL RESULT (name=<resolved>, id=<tool_call_id>) =====` blocks. The
  resolved name comes from an assistant `tool_call.id → name` map walked over the
  transcript, so each tool result is linked to the call that produced it via its
  `tool_call_id`.
- A trailing `===== ASSISTANT =====` cue marking where the model should write the
  next turn.

A lone `{"role":"user"}` message with string content and no other turns is
returned verbatim, preserving the simplest plain-text path.

**Structured output keeps `json.dumps`.** When the request carries a
`response_format`, the renderer is deliberately bypassed and `json.dumps(msg,
...)` is kept. Structured-output clients (Open WebUI and similar) depend on the
strict-JSON contract that the existing structured-output system prompt enforces;
routing those requests through the role-tagged transcript would break that
contract, so the structured-output branch is left unchanged.

## Robust extraction

`extract_tool_calls` is built to tolerate how a prompt-emulated base model
actually emits tool calls:

- **Prose before JSON** — leading narration is ignored; the function scans for the
  fenced block or a bare `{"tool_calls"...}` object anywhere in the text.
- **Fence variants** — a tolerant fenced regex captures the balanced `{...}`
  object whether the fence is ` ```json ` with a newline, a single-line fence, a
  no-`json`-label fence, or a fence with no trailing newline; the fence itself is
  excluded from the captured object.
- **Bare (un-fenced) object** — when there is no fence, a brace-balanced scan
  recovers the `{"tool_calls":...}` object directly.
- **Unescaped inner-quote arguments** — each block is first parsed with strict
  `json.loads`, then retried after unescaping `\"`→`"`, and finally recovered with
  a per-function brace-balanced fallback that pulls `arguments` out as the
  balanced `{...}` object. This recovers the common case where a model emits
  `"arguments": "{"command": "echo ..."}"` with the inner quotes unescaped.
- **No detection → plain text** — if nothing parses to a tool call, the function
  returns `None` and the request falls through to a normal text completion. It
  never invents a call.

**Speculative-planning policy.** Multiple tool calls inside ONE
`{"tool_calls":[...]}` block are all kept. Multiple SEPARATE fenced blocks are
treated as the model planning several steps ahead; by default only the FIRST
block is kept. This is gated by the environment variable
`STRAICO_TOOLCALL_KEEP_FIRST` (default `"true"`; set `"false"` to keep all
blocks). The per-call `keep_first_across_blocks` argument overrides the env when
provided.

## Documented residual limitations (the honest ceiling)

Straico has **no native tool API**. Everything above is prompt-injection emulation
plus parse-back. Reliability is greatly improved over the prior single-block
extractor, but it can NOT be guaranteed the way a native Anthropic/OpenAI tool
endpoint can. Specifically:

1. **Prose leakage.** A base model may still occasionally narrate or leak a tool
   call as prose instead of emitting the fenced JSON block. When that happens the
   extractor correctly returns `None` (no fabrication), but the intended tool call
   is lost for that turn.
2. **Literal unescaped braces in arguments.** An `arguments` value that itself
   contains a literal unescaped `{` or `}` (e.g. a shell command `echo "}"`) can
   mis-balance the brace scan; such a call may be dropped (or yield `None`) rather
   than crash. The function fails safe and never raises. The named
   unescaped-inner-quote case (`echo STEP_ONE_7`) IS recovered as valid JSON; only
   a literal brace *inside the value* hits this edge.
3. **Multi-hop depends on the client resending `tools`.** Keeping the tool channel
   open relies on the client including the `tools` array on each follow-up turn
   (the standard OpenAI multi-hop pattern). A client that drops `tools` after the
   first call disables the tool-parse gate for subsequent turns.
4. **Deferred dependency bug: aio_straico async `_reconnect`.** The vendored
   `aio_straico` async client's `_reconnect` calls httpx `.close()` on an
   `AsyncClient` (`aio_straico/async_client.py:99`) where `.aclose()` is correct
   (the package's own `aclose` at `async_client.py:425-426` does call
   `await self._session.aclose()`). This is a known reliability bug that lives in
   the dependency, outside this focused diff, and is DEFERRED here. Mitigate via
   retry-on-500 at the proxy level or by pinning httpx if it bites.

## Tests / verification

- **Deterministic unit tests** — `test/test_tool_call_extraction.py` covers the
  extractor and the transcript renderer with no network, no proxy, and no live
  model. It runs as a plain script
  (`.venv/bin/python test/test_tool_call_extraction.py`) and is also
  pytest-collectable. Cases: prose-before-toolcall, single-line / trailing fence,
  unescaped inner-quote arguments (asserting the recovered `arguments` json.loads
  cleanly to `{"command": "echo STEP_ONE_7"}`), multi-block keep-first, multiple
  calls in one block, plaintext → `None` (no fabrication), bare no-fence with
  prose, and the brace-in-value residual (asserting it never raises); plus
  transcript assertions for the single-user shortcut, system → preamble,
  `tool_call_id` linkage, and the trailing assistant continuation cue.
- **Live multi-hop acceptance** — the end-to-end 3-step sequential bash-tool
  prompt (returning the real concatenated `STEP_ONE_7, STEP_TWO_15,
  STEP_THREE_done`) is verified against the running proxy via the Hermes agent.
  This live check, being coin-costing and model-dependent, is run at verify time
  rather than in the deterministic unit suite.
