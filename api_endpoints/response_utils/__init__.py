import codecs
import re
import json
from os import environ
from json import loads

__FIX_ESCAPE_TYPOS = environ.get("FIX_ESCAPE_TYPOS", "true").strip() == "true"


def fix_escaped_characters(text_with_errors: str) -> str:
    """
    Corrects specific "double-escaped" character sequences in a string.

    For example, '\\n' becomes '\n'.

    Args:
        text_with_errors: The input string with potential double-escaped sequences.

    Returns:
        A string with specific escape sequences corrected.
    """

    if not __FIX_ESCAPE_TYPOS:
        return text_with_errors
    if text_with_errors is None:
        return ""
    # Perform specific replacements
    fixed_text = text_with_errors
    fixed_text = fixed_text.replace("\\n", "\n")  # \n -> newline
    fixed_text = fixed_text.replace("\\t", "\t")  # \t -> tab
    fixed_text = fixed_text.replace('\\"', '"')  # \" -> "
    fixed_text = fixed_text.replace("\\'", "'")  # \' -> '

    return fixed_text


def load_json_with_fixed_escape(text_with_errors: str):
    if not __FIX_ESCAPE_TYPOS:
        return loads(text_with_errors)

    try:
        return loads(text_with_errors)
    except:
        return loads(fix_escaped_characters(text_with_errors))


_FENCED_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def _balanced_object(s, start):
    """Return the brace-balanced {...} substring starting at the first '{' at/after `start`, else None."""
    i = s.find("{", start)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i:j + 1]
    return None


def _calls_from_block(block):
    # 1) strict JSON, then a tolerant retry (unescape \" -> ")
    for loader in (lambda b: json.loads(b), lambda b: json.loads(b.replace('\\"', '"'))):
        try:
            parsed = loader(block)
            if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
                out = []
                for tc in parsed["tool_calls"]:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    if fn.get("name"):
                        args = fn.get("arguments", "")
                        if not isinstance(args, str):
                            args = json.dumps(args)
                        out.append({"name": fn["name"], "arguments": args})
                if out:
                    return out
        except Exception:
            pass
    # 2) per-function brace-balanced fallback (recovers UNESCAPED inner-quote arguments)
    out = []
    for fm in re.finditer(r'"function"\s*:\s*\{', block):
        obj = _balanced_object(block, fm.start())
        if not obj:
            continue
        nm = _NAME_RE.search(obj)
        if not nm:
            continue
        ai = obj.find('"arguments"')
        args = "{}"
        if ai != -1:
            bal = _balanced_object(obj, ai)
            if bal:
                args = bal
            else:
                qm = re.search(r'"arguments"\s*:\s*"([^"]*)"', obj)
                if qm:
                    args = qm.group(1)
        out.append({"name": nm.group(1), "arguments": args})
    return out


def extract_tool_calls(text, keep_first_across_blocks=None):
    """Extract tool calls a prompt-emulated model emitted as fenced/bare JSON.

    Returns a list of {"type":"function","function":{"name","arguments"}} dicts,
    or None when no tool call is detected (NEVER fabricates a call).

    NOTE: Straico has NO native tool-calling API. Tools are emulated by injecting
    a prompt that teaches the model to emit a ```json {"tool_calls":[...]}``` block,
    which this function parses back out. Reliability is greatly improved over a
    strict json.loads (handles prose-before-JSON, single-line / no-newline fences,
    trailing fences, multiple calls, and UNESCAPED inner-quote arguments via a
    brace-balanced scan) but can NEVER be guaranteed the way a real
    Anthropic/OpenAI tool endpoint is. Known residual: an `arguments` VALUE that
    itself contains a literal unescaped brace (e.g. a shell command `echo "}"`)
    can mis-balance the brace scan; such a call may be dropped/None rather than
    crash. Multiple SEPARATE fenced blocks are treated as speculative planning and
    only the FIRST is kept (env STRAICO_TOOLCALL_KEEP_FIRST, default "true"; set
    "false" to keep all); multiple calls inside ONE block are all kept.
    """
    if keep_first_across_blocks is None:
        keep_first_across_blocks = environ.get("STRAICO_TOOLCALL_KEEP_FIRST", "true").strip().lower() == "true"
    if not isinstance(text, str) or "tool_calls" not in text:
        return None
    blocks = _FENCED_RE.findall(text)
    if not blocks:
        idx = text.find('{"tool_calls"')
        if idx == -1:
            idx = text.find('{ "tool_calls"')
        if idx != -1:
            bal = _balanced_object(text, idx)
            if bal:
                blocks = [bal]
    if not blocks:
        return None
    per_block = [c for c in (_calls_from_block(b) for b in blocks) if c]
    if not per_block:
        return None
    if len(per_block) > 1 and keep_first_across_blocks:
        calls = per_block[0]
    else:
        calls = [c for block in per_block for c in block]
    if not calls:
        return None
    return [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]


def render_chat_transcript(messages):
    """Render an OpenAI chat `messages` array as a single delimited transcript STRING.

    Straico's prompt-completion API accepts only ONE message string (no roles, no
    system param), so to stop the model reading the conversation as an *example*
    we render it as role-tagged turns it reads as its OWN history. A single user
    message with str content and no other turns is returned verbatim (preserves
    the simplest plain-text path). System messages become a leading SYSTEM block;
    assistant tool_calls are rendered back as a ```json {"tool_calls":[...]}```
    block; `tool` messages become TOOL RESULT blocks tagged with their
    tool_call_id and the resolved tool name; a trailing ASSISTANT cue tells the
    model to produce only the final assistant turn.
    """
    def _text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
            return "".join(parts)
        return ""

    if not isinstance(messages, list):
        return _text(messages)

    if (
        len(messages) == 1
        and isinstance(messages[0], dict)
        and messages[0].get("role") == "user"
        and isinstance(messages[0].get("content"), str)
    ):
        return messages[0]["content"]

    system_texts = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            text = _text(msg.get("content"))
            # The proxy's tools path stashes the client's REAL tool definitions on a
            # non-standard "tools" key of the injected system message (see
            # lm_studio/chat.py). Straico only ever sees this rendered string, so the
            # schemas MUST be emitted here — otherwise the model is blind to which
            # tools exist and their parameters, seeing only the hardcoded example in
            # the instruction text. This mirrors the pre-transcript json.dumps(msg)
            # behaviour, which serialized this key. Without it, a spec-compliant
            # OpenAI client that puts tool schemas only in the `tools` field (not in
            # its prompt) regresses vs the old code.
            tool_defs = msg.get("tools")
            if tool_defs:
                text += (
                    "\n\n## Available tools — you may ONLY call these (names + parameter schemas):\n"
                    "```json\n" + json.dumps(tool_defs, ensure_ascii=False) + "\n```"
                )
            system_texts.append(text)

    id_to_name = {}
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                    if tc.get("id"):
                        id_to_name[tc["id"]] = fn.get("name", "?")

    sections = [
        "The following is the real conversation so far (your own earlier turns "
        "included), not an example. Continue ONLY the final assistant turn.\n"
    ]

    if system_texts:
        sections.append("===== SYSTEM =====\n" + "\n".join(system_texts) + "\n")

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            continue
        if role == "user":
            sections.append("===== USER =====\n" + _text(msg.get("content")))
        elif role == "assistant":
            section = "===== ASSISTANT =====\n" + _text(msg.get("content"))
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                section += (
                    "\n```json\n"
                    + json.dumps({"tool_calls": tool_calls})
                    + "\n```"
                )
            sections.append(section)
        elif role == "tool":
            tcid = msg.get("tool_call_id")
            sections.append(
                "===== TOOL RESULT (name={}, id={}) =====\n".format(
                    id_to_name.get(tcid, "?"), tcid
                )
                + _text(msg.get("content"))
            )

    sections.append("===== ASSISTANT =====")

    body = "\n\n".join(sections[:-1])
    return body + "\n===== ASSISTANT ====="


if __name__ == "__main__":
    print(fix_escaped_characters("This should be a newline |\n|"))
    print(fix_escaped_characters("This should be a newline |\\n|"))
    print(fix_escaped_characters("This should be a tab |\t|"))
    print(fix_escaped_characters("This should be a tab |\\t|"))
    print(fix_escaped_characters("This should be a newline |\\n\\n|"))
    print(fix_escaped_characters("This should be a newline |\n\n|"))

    print(
        fix_escaped_characters(
            """
Tôi là Roo, một trợ lý kỹ thuật am hiểu, tập trung vào việc trả lời các câu hỏi và cung cấp thông tin về phát triển phần mềm, công nghệ và các chủ đề liên quan. Trong chế độ 'Hỏi' này, tôi có thể:
*   Trả lời các câu hỏi về phát triển phần mềm và công nghệ.
*   Cung cấp thông tin về các khái niệm, công cụ và phương pháp.
*   Sử dụng các công cụ để đọc tệp, tìm kiếm tệp, liệt kê tệp và định nghĩa mã.\\n\\n
*   Yêu cầu hướng dẫn cho các tác vụ cụ thể.
*   Hỏi bạn các câu hỏi tiếp theo nếu tôi cần thêm thông tin.
*   Đề xuất chuyển sang các chế độ khác (ví dụ: chế độ 'Mã' để thực hiện thay đổi mã).
Tôi sẵn sàng giúp bạn với các câu hỏi kỹ thuật của bạn.    
    """.strip()
        )
    )

    print(
        fix_escaped_characters(
            """
<thinking>
ユーザーは私が何ができるか尋ねています。私は技術アシスタントとして、ソフトウェア開発、テクノロジー、関連トピックに関する質問に答え、情報を提供することに焦点を当てています。私の能力と利用可能なツールについて説明する必要があります。
</thinking>
私はRooです。技術アシスタントとして、ソフトウェア開発、テクノロジー、関連トピックに関する質問に答え、情報を提供することに焦点を当てています。
\\n
\\t具体的には、以下のことができます。

*   **ファイルやディレクトリの操作:** ファイルの内容を読んだり、ディレクトリ内のファイルやディレクトリを一覧表示したりできます。
*   **コードの分析:** ソースコードの定義（クラス、関数など）をリストアップしたり、ファイル内の特定のパターンを検索したりできます。
*   **情報の検索:** ファイルシステム内で正規表現を使用して情報を検索できます。
*   **質問への回答:** ソフトウェア開発やテクノロジーに関する質問に答えることができます。
*   **タスクの実行:** ファイルの作成や編集など、特定のタスクを実行するためのツールを使用できます。ただし、現在の「Ask」モードではファイルの書き込みはできません。ファイルの書き込みが必要な場合は、「Code」モードなどの書き込み可能なモードに切り替える必要があります。
*   **指示の取得:** 特定のタスクを実行するための手順を取得できます。
*   **フォローアップ質問:** タスクを完了するために追加情報が必要な場合に、ユーザーに質問できます。

これらのツールと能力を組み合わせて、様々な技術的な課題を支援できます。何かお手伝いできることはありますか？    
    """.strip()
        )
    )
