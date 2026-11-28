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
