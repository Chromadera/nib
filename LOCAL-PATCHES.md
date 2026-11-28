# Local patches

This checkout is **not** vanilla upstream. Rebuilding from `main` silently drops the work below.

## `tool-choice-support` (branch, 2026-11-28)

`src/server.cpp` — honours the OpenAI `tool_choice` field, which upstream parses nowhere
(`none` still called tools, `required` was ignored):

- `none` — tool_declare block suppressed, `<|content_invoke_tool_json|>` /
  `<|content_invoke_tool_text|>` banned at sample time.
- `required` — system directive + primed `<|message_model|>`.
- `{"type":"function","function":{"name":X}}` — same directive + primed
  `<|message_model|>X<|content_invoke_tool_json|>`.

Supports `SampleParams::ban` (hard token suppression); banned requests bypass the GPU fast
sampler and MTP and force `need_logits` on.

Effect measured with `tool-eval-bench --hardmode` (104 scenarios, temp 0, seed 42,
`reasoning_effort 0.7`): one scenario changed, TC-45 fail -> pass, 182 -> 190/208, wall unchanged.

Details and the rejected alternatives: `~/Nextcloud/Projects/nib-inkling-2-mini/NOTES-tool-calling.md`.
Keep this file and the branch when rebasing onto upstream; drop both once upstream implements it.

## `response_format` JSON constrained decoding (branch, 2026-11-28)

Upstream accepts `response_format` in the request and ignores it: the field is parsed nowhere, so
`json_object` and `json_schema` produce prose. This branch implements constrained decoding:

- `src/json_constraint.hpp` (new) — byte-level pushdown automaton over canonical JSON. Supports
  `type` (object/array/string/integer/number/boolean/null), `properties`, `required`, `items`,
  string `enum`. A schema object with no declared keys means "unconstrained object" (any keys),
  which is what `json_object` maps to. Anything unsupported (`$ref`, `anyOf`/`oneOf`, `pattern`,
  `min/max`) degrades to "any JSON value" instead of mis-constraining.
- `src/server.cpp` — builds the machine from the request, then masks per sampling step through
  `SampleParams::allow`; `cst_before_sample` / `cst_after_sample` are the only two integration
  points, called from all three decode paths (solo fast path, batched B==1, batched B>=2).
  Constrained requests bypass the GPU fast sampler and MTP and force `need_logits`.
- `tests/test_json_constraint.cpp` — 26 assertions, standalone `g++ -std=c++17`, no HIP.

Behaviour decisions that are not obvious:

- The constraint **arms on `<|content_text|>` and disarms on the end-of-message token**, so the
  thinking block stays free-form. Masking from token 0 blocks all reasoning.
- Specials are **blocked mid-value**. Allowing them lets the model escape the grammar: it emitted
  `{` and then `<|end_message|>`, and the caller got a single brace.
- After the value completes, only specials are allowed, otherwise the model appends prose
  (`{} Lisbon is…`).
- Keys are emitted in schema order (declaration order) and objects must carry at least one
  non-empty-keyed member. Whitespace between tokens is allowed: rejecting it makes a model that
  writes pretty JSON contort around the mask (observed: an empty `""` key and a dropped property).
  The machine guarantees a *valid* document matching the schema, not a canonical layout.
- Unconstrained (`json_object`) objects can sprawl; a caller that asks for many keys needs a
  `max_tokens` budget large enough for the thinking block too, or the value truncates mid-token.

Verified end to end through llama-swap: typed object enforcing three properties, string enum,
array of strings, non-empty `json_object`, no trailing prose, thinking still free-form, malformed
`json_schema` -> 400, plain requests unchanged.

Effect measured with `tool-eval-bench --hardmode` (104 scenarios, temp 0, seed 42,
`reasoning_effort 0.7`): score 91 -> 91, 190 -> 190 points,
deployability 90 -> 90, responsiveness 87 -> 87, **zero scenarios changed status**. The bench does
send `response_format` on structured-output scenarios (TC-65), which is what caught the bugs above.
