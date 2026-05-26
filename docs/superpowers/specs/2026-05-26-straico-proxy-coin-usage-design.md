# Surface Straico coin price under `usage.straico_coins`

**Status:** Approved (design phase) · 2026-05-26
**Branch:** `feature/return-real-usage-from-straico`
**Builds on:** `e01364b` (return real word-count usage from straico instead of hardcoded placeholder)
**PR target:** fork-only (`happyhackerbird/ollama-straico-apiproxy`); not upstream

## Goal

Surface Straico's actual coin price in the proxy's OpenAI-compat
`/v1/chat/completions` response so downstream clients can track real spend in
the only billing currency that maps 1:1 with the user's Straico account
ledger (coins), not approximations (words-mapped-as-tokens × upstream-API
list price → notional USD).

## Why this matters

The `e01364b` commit fixed the hardcoded-usage bug by extracting `words` from
Straico's response and surfacing them as `prompt_tokens` / `completion_tokens`.
That's an improvement, but it stops short:

- **Words ≠ tokens.** The commit message itself flags 2-4× undercount on
  JSON-heavy prompts (which is exactly what code/structured-output clients
  send).
- **Token rates × upstream-API list prices ≠ Straico cost.** Straico is a
  bundled-subscription provider. Multiplying token approximations by
  Anthropic / OpenAI list prices measures "what this would cost on the
  direct API" — not what the user is actually being charged in their
  Straico account.

The actual debit is in Straico's `price` field, denominated in coins. It's
already present in every prompt-completion response. The patch in `e01364b`
extracted `words` and left `price` on the floor.

## Evidence: Straico response shape

Empirically verified 2026-05-26 against the live Straico API via the local
proxy's `aio_straico` client. Probe script at `/tmp/straico_probe_2026-05-26.py`.

The v0 prompt-completion response (single-model, no-image path — the only
path scope-agent's text-only pipeline triggers) has this shape:

```json
{
  "completion": { "choices": [ { "message": {...} } ] },
  "price": { "input": 0.16, "output": 0.04, "total": 0.20 },
  "words": { "input": 4, "output": 1, "total": 5 }
}
```

Verified across three model families and two input sizes:

| Model                  | Input words | Output words | Price (coins)              |
|------------------------|-------------|--------------|----------------------------|
| `claude-haiku-4-5-5`   | 4           | 1            | input=0.16 out=0.04 t=0.20 |
| `openai/gpt-5-nano`    | 4           | 1            | input=0.02 out=0.01 t=0.03 |
| `amazon/nova-lite-v1`  | 4           | 1            | input=0.01 out=0    t=0.01 |
| `claude-haiku-4-5-5`   | 124         | 1            | input=4.96 out=0.04 t=5.00 |

Observations:
- `price` is **always present** for successful completions.
- `price.total` scales with input size (25× more words → 25× more coins
  at fixed model), confirming this is the actual per-call debit.
- Per-model rate is set by Straico and differs across families — exactly
  the data scope-agent's current flat-per-call `PER_CALL_COST_USD` table
  is trying to approximate.

The v1 path (`response["completions"][model]["completion"]`, triggered only
when images are attached — see `backend/straico.py:236-252`) was not probed.
By symmetry with `words` → `overall_words` in `e01364b`, the design assumes
`overall_price` is the v1 equivalent and tolerates either name. Scope-agent
does not use the image path, so this is defensive coverage only.

## Design

### Field placement: `usage.straico_coins`

Extend the existing `usage` dict (returned by `_extract_usage`) with a new
vendor-extension subfield:

```json
"usage": {
  "prompt_tokens": 4,
  "completion_tokens": 1,
  "total_tokens": 5,
  "straico_coins": {
    "input": 0.16,
    "output": 0.04,
    "total": 0.20
  }
}
```

### Alternatives considered and rejected

**Alt 2 — Top-level `straico` field**, splitting billing data between
`usage` (words) and `straico` (coins). Rejected because cost data belongs
with cost data; splitting forces consumers into two read paths for one
concept.

**Alt 3 — Pass through Straico's `price`/`words` raw at top level.**
Rejected — collides with the OpenAI response shape (no top-level `price`
in the standard); breaks strict clients.

### Why `usage.straico_coins` is sound

- Co-located with the existing usage shape; one attribute path for the
  consumer: `response.usage["straico_coins"]["total"]`.
- Precedent for `usage` extensions exists in the OpenAI ecosystem
  (Anthropic adds `cache_creation_input_tokens` under `usage`; OpenAI
  adds `prompt_tokens_details`).
- Explicit unit in the field name — `straico_coins` makes it
  unambiguous that this is not tokens, not USD, but Straico's
  billing-ledger unit.
- OpenAI-compat clients tolerate unknown keys in `usage` (verified in
  the wild — clients deserialize what they recognize, ignore the rest).

## Implementation

### `backend/straico.py` — extend `_extract_usage`

The existing function (added in `e01364b`) extracts words. The change
adds independent fail-soft extraction of `price`:

```python
def _extract_usage(response):
    if not isinstance(response, dict):
        return None
    words = response.get("overall_words") or response.get("words")
    if not isinstance(words, dict):
        return None
    try:
        usage = {
            "prompt_tokens": int(words.get("input", 0)),
            "completion_tokens": int(words.get("output", 0)),
            "total_tokens": int(words.get("total", 0)),
        }
    except (TypeError, ValueError):
        return None
    # Surface Straico's coin debit when present. Independent fail-soft so
    # a missing/malformed price never invalidates word-count usage.
    price = response.get("overall_price") or response.get("price")
    if isinstance(price, dict):
        try:
            usage["straico_coins"] = {
                "input": float(price.get("input", 0)),
                "output": float(price.get("output", 0)),
                "total": float(price.get("total", 0)),
            }
        except (TypeError, ValueError):
            pass
    return usage
```

### `api_endpoints/lm_studio/response/basic/completion_response.py`

No change. It already receives the `usage` dict from `_extract_usage` and
serializes it onto the response. The new `straico_coins` subfield rides
along automatically.

### Tests

`_extract_usage` has no unit tests today (`e01364b` shipped without them).
The existing `test/` directory contains integration tests (pytest, hitting
a live proxy at `127.0.0.1:3214` — see `test/test_ollama.py` for the
pattern). `backend/test.py` is a backend stub used to swap out
`backend/straico.py` during integration runs; it is not a unit-test
harness for `_extract_usage`.

This PR adds a new unit-test file `test/test_extract_usage.py` that
imports `_extract_usage` directly from `backend.straico` and exercises
the pure-function behavior with mock dict inputs (no proxy server
required). Cases:

- (a) response with both `words` and `price` → both surface, including
  per-direction coin split
- (b) response with `words` only (`price` missing) → `straico_coins`
  absent, word usage works
- (c) response with malformed `price` (e.g., string instead of dict;
  `input` non-numeric) → `straico_coins` absent, word usage works,
  no exception raised
- (d) response with neither `words` nor `price` → returns `None`
  (preserves prior behavior from `e01364b`)
- (e) response with `overall_words` and `overall_price` (v1 shape) → both
  surface under the standard field names

Backfilling unit coverage for the `e01364b` word-extraction logic falls
out of these cases for free — case (b) and (d) cover its behavior,
closing a small test debt as a side effect.

### Fail-soft layering rationale

Words and coins are extracted independently because their failure modes
are independent. A future Straico API change that renames `price` should
not take down word-count usage that's already shipping. Inverted: if
`words` is malformed, the function returns `None` and the OpenAI-compat
layer falls back to its prior behavior — surfacing coins alone without
word context would be confusing in the OpenAI response shape.

## Data flow (end-to-end)

```
Straico API
    ↓ {completion, price:{input,output,total}, words:{input,output,total}}
backend.straico.prompt_completion
    ↓ (content, reasoning, _extract_usage(response))
api_endpoints.lm_studio.chat
    ↓ basic_response(..., usage=usage)
HTTP /v1/chat/completions
    ↓ usage:{prompt_tokens, completion_tokens, total_tokens, straico_coins:{...}}
[downstream consumer, e.g. scope-agent]
    reads usage["straico_coins"]["total"]
```

The proxy PR's contract ends at "client can read
`response.usage.straico_coins`". Consumer-side cost tracking is a
separate PR in the consumer repo.

## Commit narrative on the feature branch

Keep `e01364b` as-is (good standalone commit on words). Add one new
commit on top:

> **surface straico coin price under usage.straico_coins**
>
> Builds on the words-as-tokens patch by also extracting `price` from
> Straico's prompt-completion response. `price` is denominated in coins
> (Straico's billing currency, distinct from upstream API USD pricing),
> and is the only field that maps 1:1 with the user's Straico account
> ledger. Surfaced as a vendor-extension subfield on `usage` so
> OpenAI-compat clients can track real cost without inventing a separate
> billing channel.
>
> Independent fail-soft: malformed/missing `price` does not invalidate
> word-count usage. Both `price` (v0) and `overall_price` (v1, untested
> but symmetric with `overall_words`) are accepted.
>
> Scope unchanged from prior commit: lm_studio non-streaming
> /v1/chat/completions only.

## Out of scope (deferred to separate PRs)

- Streaming response (`response/stream/completion_response.py`)
- Tool-call inline responses in `lm_studio/chat.py`
- ollama / claude / agent / embedding emulation paths
- Build + publish workflow under
  `ghcr.io/happyhackerbird/ollama-straico-apiproxy:latest`
- Consumer-side switch to coin-based capping (scope-agent's `spend.py`
  refactor to `record_coins` + `DAILY_SPEND_CAP_COINS`)

## Success criteria

The PR is complete when:

1. `_extract_usage` returns `usage["straico_coins"]` whenever the
   underlying Straico response carried a usable `price` / `overall_price`
   dict.
2. The four `_extract_usage` test cases above pass.
3. Manually verified end-to-end: a probe against the running proxy after
   merge returns `usage.straico_coins.total > 0` for at least one model,
   and the value matches the Straico response's `price.total` within
   floating-point tolerance.
4. The proxy's `/v1/chat/completions` response shape remains
   backward-compatible: existing fields unchanged, only a new subfield
   added to `usage` (and only when data is available).
