# Third-party projects, data and services

MIT (see `LICENSE`) covers this harness and the 72 original public decisions in
`datasets/public/original.jsonl`. Nothing else in this list is ours to license.

## Systems evaluated

Each was reached through the interface its author published. We used public
endpoints as ordinary clients, at one request at a time, and never sent a
provider's API key to anyone else's endpoint.

| System | Author | Source |
|---|---|---|
| Jev 1.13.0 | TypeSafe AI | <https://docs.typesafe.ai> |
| openjev-sglang | ekzhang | <https://github.com/ekzhang/openjev-sglang> |
| system-one-open | mithalouni | <https://github.com/mithalouni/system-one-open> |
| open-alternative-jev | IkerMoel | <https://github.com/ikermoel/open-alternative-jev>, <https://huggingface.co/spaces/IkerMoel/open-alternative-jev> |
| open-jev-deberta-v3-large | Kotoba Labs | <https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large>, <https://github.com/kotoba-lang/typed-decisions> |
| GPT-5.6 Luna | OpenAI | <https://platform.openai.com> |
| Gemini 3.1 Flash-Lite | Google | <https://ai.google.dev> |
| DeepSeek V4.1 Flash | DeepSeek | <https://api-docs.deepseek.com> |
| Qwen3.8 27B | Qwen, served by Chutes | <https://chutes.ai> |

Model weights, base models and each project's own code keep their own licences.
A permissive licence on a repository is not a licence for the base model it
fine-tunes, and we do not restate either.

## Wire format

The typed-decision request shape (`state`, `questions`, the `noul` / `choice` /
`score` primitives) is TypeSafe's public HTTP API, documented at
<https://docs.typesafe.ai/api>. Several of the open rebuilds implement it
deliberately, which is why one adapter reaches more than one of them. JevBench is
not affiliated with or endorsed by TypeSafe AI.

## Imported decisions

146 of the 242 decisions come from our own earlier auto-router experiment
(<https://github.com/fstandhartinger/auto-model-router>): 78 routing requests and
68 answer-adequacy judgements whose ground truth is a deterministic grader's
verdict on a saved answer. The upstream task text is **not** redistributed here,
because the datasets it was drawn from keep their own terms. `datasets/manifest.json`
pins the hashes; `scripts/import_router.py` shows exactly what was taken and what
was excluded.

## Held-out decisions

24 scenarios are written by us and deliberately unpublished, so the suite cannot
be trained on in full. Only their whole-split hash and aggregate results appear
here. They are sent to the services being evaluated, which is not the same thing
as being public - see the limits section of the README.
