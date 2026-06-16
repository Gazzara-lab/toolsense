# Reproduction notes — evaluation harness

This document records (1) what the `evaluate/` harness measures and how it relates
to the paper, and (2) dataset-integrity facts that can be re-derived offline from
the shipped JSONL files with zero API calls.

## What the harness measures (and what it does not)

The `evaluate/` package scores any LiteLLM model on the three shipped diagnostic
datasets:

| Command | Dataset | Metric | Random baseline |
|---|---|---|---|
| `eval-qa`  | `data/toolsense-qa/data.jsonl` | yes/no accuracy | 50% |
| `eval-mcq` | `data/toolsense-mcq/data.jsonl` | 4-way accuracy | 25% |
| `eval-rrb` | `data/toolsense-realistic-retrieval/data.jsonl` | Recall@k / hit@k / MRR / nDCG@k over the 14-tool pool | — |

**Scope / honesty note.** This is an *inference-only, in-context* evaluation:

- For QA/MCQ, a generic API model has no trained virtual tool token, so the
  phrase "this tool" is grounded with a textual referent. By default
  (`--tool-ref name`) only the tool name + title are shown — this probes what the
  model knows about the named tool. `--tool-ref description` adds the description
  (a reading-comprehension upper bound); `--tool-ref none` reveals nothing (a
  blind lower bound). Because RapidAPI tool names are often descriptive, expect
  `name` mode to be easy when the name encodes the answer; interpret accordingly.
- For RRB, the model ranks the **shipped 14-tool `analyzed_tools` candidate pool**
  in-context. This is **not** a reproduction of the paper's headline `Rc@50`,
  which is trie-constrained beam search over the full ~47k-tool catalog produced
  by a *trained* ToolGen retriever. The numbers are therefore not directly
  comparable to the paper's tables; `k` ranges over the pool size (≤ 14).
- Output is prompt-constrained ("answer with one letter/word"; "return the ranked
  list"), not logit-constrained as in the paper. Responses are parsed robustly;
  unparseable answers are counted and reported.

The harness adds the missing *scorer*; it does not retrain models or reproduce the
paper's trained-retriever numbers.

## Response parsing (robustness and its limits)

Models do not always obey "answer with one word/letter" — they explain, use
markdown, restate the options, or answer with a synonym. The parsers in
`evaluate/common.py` are hardened against the common cases and verified by
known-answer tests in `evaluate/tests/test_metrics.py`:

- **QA** — an explicitly *declared* answer wins, and the last declaration wins, so
  "the older version said yes, so no" is read as **No**. A clean leading answer is
  unaffected ("Yes, although there is no `no` evidence …" → Yes). Bare synonyms at
  the start ("Nope", "Yeah", "Y"/"N", "True"/"False") are recognized only as a last
  resort.
- **MCQ** — the chosen letter is the last explicit decision ("… I pick D"); a
  *restated* option list ("the options are A, B, C, D …") is not mistaken for the
  answer. Letters are matched uppercase-only, so the article "a" and prose words are
  never read as a choice.
- **RRB** — a ranked-list reply (`1. Tool 7 / 2. Tool 14 / …`) has its leading rank
  numbers stripped so the real tool ids are scored; a compliant comma list is left
  untouched. Decimal scores ("0.95") never leak stray indices.

A guess is never preferred over an honest non-answer: genuinely ambiguous or
non-compliant output (self-corrections like "no, wait, yes"; non-English answers;
free-form prose rankings with digits embedded in tool names) is reported as
unparseable / scored on what was parsed, and the **unparseable count is always
printed** so non-compliance is visible rather than silently mis-scored. For
trustworthy numbers, evaluate instruction-following models and check that count.

## Example run (real model)

A reference run on the full datasets with `google/gemma-3-12b-it`
(`temperature=0`), via an OpenAI-compatible proxy. Gemma 3 is one of the backbone
families the paper studies, but note this is a generic in-context API call — the
model has no trained virtual tool token, so the numbers are **not** comparable to
the paper's trained-ToolGen results (see the scope note above). Re-run with the
commands in the README, sweeping `--tool-ref name|description|none`.

**Reproducibility.** The *scoring* is fully deterministic — candidate shuffles are
seeded by record id (independent of `PYTHONHASHSEED`) and the bootstrap CIs use
`seed=0`, so identical predictions yield identical tables and `--dry-run` is
byte-for-byte reproducible. The *model* side is not: the endpoint is called at
`temperature=0` but with no `seed` and no provider pin, and `google/gemma-3-12b-it`
is an OpenRouter alias rather than a pinned snapshot (the alias may route to
different backends over time, greedy decoding is not bit-reproducible across
providers/hardware, and the weights can be updated). So treat these as
**illustrative numbers, reproducible up to model/provider nondeterminism** — a
re-run should land close but need not match digit-for-digit. The persisted
`*_predictions.jsonl` are the exact reproducible record; re-scoring them reproduces
the tables and CIs precisely.

Point estimates are shown with 95% percentile-bootstrap CIs (`n_resamples=1000`,
`seed=0`), matching the form the paper reports throughout.

**QA / MCQ probing accuracy** (0 unparseable responses in every run):

| `--tool-ref` | QA (n=500, rand 0.50) | MCQ (n=496, rand 0.25) |
|---|---|---|
| `none` (identity hidden) | 0.632 [0.592, 0.674] | 0.587 [0.544, 0.629] |
| `name` (name + title) | 0.856 [0.824, 0.886] | 0.948 [0.925, 0.966] |
| `description` (+ description) | 0.996 [0.990, 1.000] | 1.000 [1.000, 1.000] |

**Tool-knowledge lift over the `none` baseline** (paired bootstrap on shared record
ids; all intervals exclude 0, so the gains are significant):

| condition − none | QA | MCQ |
|---|---|---|
| `name` − `none` | +0.224 [0.168, 0.276] | +0.361 [0.319, 0.407] |
| `description` − `none` | +0.364 [0.324, 0.408] | +0.413 [0.371, 0.460] |

**Realistic Retrieval** (rank the shipped 14-tool pool; n=500):

| tier | n | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|
| overall | 500 | 0.285 | 0.545 | 0.692 | 0.895 | 0.725 |
| easy | 167 | 0.485 | 0.731 | 0.820 | 0.940 | 0.629 |
| medium | 167 | 0.249 | 0.568 | 0.730 | 0.920 | 0.770 |
| hard | 166 | 0.120 | 0.335 | 0.524 | 0.824 | 0.777 |

Overall 95% CIs: R@1 [0.254, 0.314], R@3 [0.512, 0.577], R@5 [0.664, 0.720],
R@10 [0.877, 0.911], MRR [0.699, 0.754].

Reading the numbers:

- **Monotonic in revealed identity.** Accuracy rises `none` < `name` <
  `description` for both QA and MCQ — the expected ordering, and a check that the
  referent plumbing and scoring are correct. With the full description in context
  the task is near-trivial (≈ ceiling); with only the name it is easier than the
  `none` floor because RapidAPI names are descriptive.
- **`none` is not at the random baseline.** With the tool fully hidden the model
  still scores 0.632 (QA) and 0.587 (MCQ), above 0.50 / 0.25. The questions and
  distractors themselves carry tool-independent signal, so the meaningful
  "tool-knowledge" quantity is the *lift* of `name` / `description` over `none`
  (quantified in the lift table above, every CI excluding 0), not the raw
  accuracy. Reporting `none` makes that controllable.
- **Retrieval degrades with complexity.** R@1 falls 0.485 (easy) → 0.249 (medium)
  → 0.120 (hard) as the gold set grows, while R@10 stays high (≥ 0.82) — the model
  ranks *a* relevant tool highly but spreads multi-tool gold sets across the list.
- **Parsing held up on real output.** Across ~3,000 generations the unparseable
  count was 0, and every recorded prediction re-derives exactly from its raw text.

This is a single-model, single-run (`temperature=0`) illustration that the harness
produces sensible, interpretable numbers — not a benchmark measurement or a
reproduction of any paper result.

## Zero-cost verification (`--dry-run`)

Every command accepts `--dry-run` (alias `--mock`), which replaces model calls
with a deterministic stub responder — no network, no API key, no paid calls, and
no heavy dependencies (the mock path uses only the standard library). This is what
CI and reviewers can run.

Running the stub over the full datasets lands near the random baselines (the stub
is gold-independent, so it only approximates them), which confirms the scoring is
wired correctly:

```
eval-qa  --dry-run : accuracy 0.526   (random 0.500)   unparseable 0
eval-mcq --dry-run : accuracy 0.248   (random 0.250)   unparseable 0
eval-rrb --dry-run : R@1 0.095  R@3 0.223  R@5 0.357  R@10 0.718  MRR 0.432
```

Metric and parser correctness is covered by known-answer unit tests in
`evaluate/tests/test_metrics.py` (run with `pytest evaluate/tests`).

## Dataset-integrity facts (verified offline)

Re-derivable from the committed JSONL files (counts via JSON line-parse, the
authoritative record count). The QA/MCQ/RRB files are newline-terminated, so
`wc -l` matches their record counts; only the tool catalog lacks a final newline,
so `wc -l data/toolbench-tools/data.jsonl` reports 46979 vs 46,980 records.

**QA — `data/toolsense-qa/data.jsonl`**
- 500 records; every record has `id`, `question`, `answer`, `tool`.
- Answer balance: 250 Yes / 250 No (exactly balanced).

**MCQ — `data/toolsense-mcq/data.jsonl`**
- 496 records; every record has exactly 3 distractors (`wrong_answers`).
- 0 records leak the correct answer into the distractors; all 4 options are
  distinct in every record.

**Realistic Retrieval — `data/toolsense-realistic-retrieval/data.jsonl`**
- 500 records; tiers: 167 easy / 167 medium / 166 hard.
- Candidate pool (`analyzed_tools`) size is exactly 14 for every record.
- Gold set sizes match the documented tier contract: easy = 1 (167×);
  medium = 2 (46×) or 3 (121×); hard = 4–8 (28× / 74× / 46× / 16× / 2×).
- The `tool` field is polymorphic: a single dict for easy, a list of dicts for
  medium/hard. `evaluate.common.gold_tool_names` normalizes both.
- All gold tools are present in the candidate pool for **500/500** records (mean
  fraction = 1.0), so pool-based Recall@k is well defined. No duplicate gold names.

**Tool catalog — `data/toolbench-tools/data.jsonl`**
- 46,980 tools; 46,980 unique `tool_name` values (0 duplicates).
