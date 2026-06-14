"""Deterministic unit tests for the Straico prompt-emulated tool-call extractor
and the role-mapping transcript renderer (U3 / design D6).

NO network, NO proxy server, NO live model. Pure dict/str-in, value-out tests
over `extract_tool_calls` and `render_chat_transcript` from
`api_endpoints.response_utils`.

Runs two ways:

  1. As a PLAIN SCRIPT (the project has no pytest dependency):

        .venv/bin/python test/test_tool_call_extraction.py

     -> runs every assertion, prints "ALL TESTS PASSED" and exits 0 on success;
        on the first AssertionError prints the failure and exits 1.

  2. Under pytest IF it happens to be installed (the test_* functions below are
     collectable), but pytest is NOT required.

Covers (D6):
  extract_tool_calls — prose-before, trailing/single-line fence, UNESCAPED
  inner-quote arguments (the key case), multi-block keep-first, multiple calls
  in one block, plaintext -> None (no fabrication), bare no-fence with prose,
  brace-in-value residual (must not crash).
  render_chat_transcript — single-user shortcut, system -> preamble,
  tool_call_id linkage, assistant continuation cue.
"""
import json
import os
import sys

# Add the repo root to sys.path so this runs as a standalone script from anywhere
# (no app bootstrap, no network) and imports the pure helpers directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from api_endpoints.response_utils import extract_tool_calls, render_chat_transcript


# --------------------------------------------------------------------------- #
# extract_tool_calls
# --------------------------------------------------------------------------- #

def test_prose_before_toolcall():
    """1. Prose, then a fenced ```json {"tool_calls":[...]}``` block -> 1 call."""
    text = (
        "Let me check the weather for you.\n"
        "```json\n"
        '{"tool_calls":[{"type":"function","function":'
        '{"name":"get_weather","arguments":"{\\"location\\":\\"Paris\\"}"}}]}\n'
        "```"
    )
    calls = extract_tool_calls(text)
    assert calls is not None, "prose-before-toolcall: expected a call, got None"
    assert len(calls) == 1, "prose-before-toolcall: expected exactly 1 call, got %r" % (calls,)
    assert calls[0]["function"]["name"] == "get_weather", (
        "prose-before-toolcall: wrong name %r" % (calls[0]["function"]["name"],)
    )


def test_trailing_single_line_fence():
    """2. A ```json {...}``` fence all on one line -> still extracts."""
    text = (
        '```json {"tool_calls":[{"type":"function","function":'
        '{"name":"terminal","arguments":"{}"}}]} ```'
    )
    calls = extract_tool_calls(text)
    assert calls is not None, "single-line-fence: expected a call, got None"
    assert len(calls) == 1, "single-line-fence: expected 1 call, got %r" % (calls,)
    assert calls[0]["function"]["name"] == "terminal", (
        "single-line-fence: wrong name %r" % (calls[0]["function"]["name"],)
    )


def test_unescaped_arguments():
    """3. THE key case: arguments with UNESCAPED inner quotes are recovered as
    valid JSON. The model emits  "arguments": "{"command": "echo STEP_ONE_7"}"
    (inner quotes not escaped) -> recovered arguments must json.loads cleanly.
    """
    # Literal unescaped inner quotes inside the arguments string value.
    text = (
        "```json\n"
        '{"tool_calls":[{"type":"function","function":'
        '{"name":"terminal","arguments":"{"command": "echo STEP_ONE_7"}"}}]}\n'
        "```"
    )
    calls = extract_tool_calls(text)
    assert calls is not None, "unescaped-arguments: expected a call, got None"
    assert len(calls) == 1, "unescaped-arguments: expected 1 call, got %r" % (calls,)
    assert calls[0]["function"]["name"] == "terminal", (
        "unescaped-arguments: wrong name %r" % (calls[0]["function"]["name"],)
    )
    args = calls[0]["function"]["arguments"]
    parsed = json.loads(args)  # must NOT raise
    assert parsed == {"command": "echo STEP_ONE_7"}, (
        "unescaped-arguments: arguments json.loads -> %r, want %r"
        % (parsed, {"command": "echo STEP_ONE_7"})
    )


def test_multi_block_keep_first_default():
    """4. Two SEPARATE fenced blocks (echo A, echo B). Default policy keeps the
    first block only (speculative-planning guard) -> exactly 1 call (echo A).
    """
    text = (
        "```json\n"
        '{"tool_calls":[{"type":"function","function":'
        '{"name":"terminal","arguments":"{\\"command\\":\\"echo A\\"}"}}]}\n'
        "```\n"
        "and then\n"
        "```json\n"
        '{"tool_calls":[{"type":"function","function":'
        '{"name":"terminal","arguments":"{\\"command\\":\\"echo B\\"}"}}]}\n'
        "```"
    )
    # Pin the default explicitly so an env override cannot flip the assertion.
    calls = extract_tool_calls(text, keep_first_across_blocks=True)
    assert calls is not None, "multi-block-keep-first: expected a call, got None"
    assert len(calls) == 1, (
        "multi-block-keep-first: expected exactly 1 call, got %r" % (calls,)
    )
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"command": "echo A"}, (
        "multi-block-keep-first: expected the FIRST block (echo A), got %r" % (args,)
    )


def test_multi_in_one_block():
    """5. One block carrying two tool calls -> both kept (returns 2)."""
    text = (
        "```json\n"
        '{"tool_calls":['
        '{"type":"function","function":{"name":"alpha","arguments":"{}"}},'
        '{"type":"function","function":{"name":"beta","arguments":"{}"}}'
        "]}\n"
        "```"
    )
    calls = extract_tool_calls(text, keep_first_across_blocks=True)
    assert calls is not None, "multi-in-one-block: expected calls, got None"
    assert len(calls) == 2, (
        "multi-in-one-block: expected exactly 2 calls, got %r" % (calls,)
    )
    names = [c["function"]["name"] for c in calls]
    assert names == ["alpha", "beta"], (
        "multi-in-one-block: wrong names %r" % (names,)
    )


def test_plaintext_returns_none():
    """6. Plain text with no tool-call envelope -> None (NO fabrication)."""
    text = "STEP_ONE_7, STEP_TWO_15, STEP_THREE_done"
    calls = extract_tool_calls(text)
    assert calls is None, (
        "plaintext->None: expected None (no fabrication), got %r" % (calls,)
    )


def test_bare_no_fence_with_prose():
    """7. Prose + a BARE (un-fenced) {"tool_calls":[...]} object -> extracts."""
    text = (
        "Sure, I will run it now "
        '{"tool_calls":[{"type":"function","function":'
        '{"name":"terminal","arguments":"{}"}}]} '
        "and report back."
    )
    calls = extract_tool_calls(text)
    assert calls is not None, "bare-no-fence: expected a call, got None"
    assert len(calls) == 1, "bare-no-fence: expected 1 call, got %r" % (calls,)
    assert calls[0]["function"]["name"] == "terminal", (
        "bare-no-fence: wrong name %r" % (calls[0]["function"]["name"],)
    )


def test_brace_in_value_residual_no_crash():
    """8. An arguments VALUE containing a literal '}' (e.g. shell echo "}") is a
    known residual: the brace scan may mis-balance. The contract is only that the
    call MUST NOT RAISE — the return may be a (possibly malformed) list or None.
    """
    text = (
        "```json\n"
        '{"tool_calls":[{"type":"function","function":'
        '{"name":"terminal","arguments":"{"command": "echo \\"}\\""}"}}]}\n'
        "```"
    )
    try:
        result = extract_tool_calls(text)
    except Exception as exc:  # pragma: no cover - the whole point is this never fires
        raise AssertionError(
            "brace-in-value: extractor raised %s: %s (must fail safe, never raise)"
            % (type(exc).__name__, exc)
        )
    # No assertion on the value: list OR None are both acceptable per the
    # locked design's documented residual. Reaching here without an exception
    # is the pass condition.
    assert result is None or isinstance(result, list), (
        "brace-in-value: expected None or a list, got %r" % (result,)
    )


# --------------------------------------------------------------------------- #
# render_chat_transcript
# --------------------------------------------------------------------------- #

def test_single_user_shortcut():
    """9. A lone user message with str content -> returned verbatim."""
    out = render_chat_transcript([{"role": "user", "content": "hello"}])
    assert out == "hello", "single-user-shortcut: expected 'hello', got %r" % (out,)


def test_system_to_preamble():
    """10. A system message -> a leading SYSTEM section carrying its text."""
    out = render_chat_transcript([
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "hi"},
    ])
    assert "===== SYSTEM =====" in out, (
        "system->preamble: missing SYSTEM delimiter in %r" % (out,)
    )
    assert "You are a careful assistant." in out, (
        "system->preamble: missing system text in %r" % (out,)
    )


def test_tool_call_id_linkage():
    """11. assistant tool_calls (id 'abc', name 'terminal') + tool result
    (tool_call_id 'abc', content 'OUT') -> TOOL RESULT block linking both.
    """
    out = render_chat_transcript([
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "abc",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "abc", "content": "OUT"},
    ])
    assert "TOOL RESULT" in out, "tool-linkage: missing 'TOOL RESULT' in %r" % (out,)
    assert "abc" in out, "tool-linkage: missing tool_call_id 'abc' in %r" % (out,)
    assert "terminal" in out, "tool-linkage: missing resolved name 'terminal' in %r" % (out,)
    assert "OUT" in out, "tool-linkage: missing tool result content 'OUT' in %r" % (out,)


def test_assistant_continuation_cue():
    """12. A multi-message transcript ends with the trailing ASSISTANT cue."""
    out = render_chat_transcript([
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ])
    assert out.rstrip().endswith("===== ASSISTANT ====="), (
        "assistant-cue: transcript must end with '===== ASSISTANT =====', tail=%r"
        % (out[-40:],)
    )


# --------------------------------------------------------------------------- #
# Plain-script runner (no pytest required)
# --------------------------------------------------------------------------- #

_TESTS = [
    test_prose_before_toolcall,
    test_trailing_single_line_fence,
    test_unescaped_arguments,
    test_multi_block_keep_first_default,
    test_multi_in_one_block,
    test_plaintext_returns_none,
    test_bare_no_fence_with_prose,
    test_brace_in_value_residual_no_crash,
    test_single_user_shortcut,
    test_system_to_preamble,
    test_tool_call_id_linkage,
    test_assistant_continuation_cue,
]


def main():
    for test in _TESTS:
        try:
            test()
        except AssertionError as exc:
            print("FAILED: %s" % (test.__name__,))
            print("  %s" % (exc,))
            sys.exit(1)
        print("ok: %s" % (test.__name__,))
    print("ALL TESTS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
