"""Evaluate a model on the ToolSense MCQ probing benchmark (4-way).

For each record the four options (1 correct + 3 distractors) are shuffled
deterministically (seeded by the record id, so runs are reproducible), presented
as A/B/C/D, and the model must pick one letter. We report exact-match accuracy
against the 25% random baseline.

Usage:
    python -m toolsense_eval.eval_mcq --data data/toolsense-mcq/data.jsonl \\
        --output results/mcq --model claude-4.5-sonnet

    # Zero-cost end-to-end check:
    python -m toolsense_eval.eval_mcq --data data/toolsense-mcq/data.jsonl \\
        --output results/mcq --dry-run
"""

from __future__ import annotations

import argparse
import os

from toolsense_eval import common, metrics, model_client

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "")

_PROMPT = """\
You are evaluating tool knowledge. Consider the following tool.

{referent}

Question: {question}
Options:
{options}

Respond with exactly one letter: A, B, C, or D. Do not explain."""

_LETTERS = ["A", "B", "C", "D"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True,
                        help="MCQ benchmark JSONL (e.g. data/toolsense-mcq/data.jsonl)")
    parser.add_argument("--output", required=True, help="Output directory for predictions + results card")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LiteLLM model string (default: {DEFAULT_MODEL!r})")
    parser.add_argument("--tool-ref", choices=["name", "description", "none"], default="name",
                        help="What to reveal as the referent of 'this tool' (default: name)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Max records to evaluate (default: all)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--dry-run", "--mock", dest="dry_run", action="store_true",
                        help="Score against a deterministic stub responder with no API calls")
    args = parser.parse_args()

    if not args.dry_run:
        model_client.load_env()
        if not args.model:
            parser.error("--model is required unless --dry-run is set (or set DEFAULT_MODEL)")

    records = common.load_jsonl(args.data)
    if args.num_samples is not None:
        records = records[: args.num_samples]
    print(f"Loaded {len(records)} MCQ records from {args.data}")
    print(f"Model: {'<mock>' if args.dry_run else args.model}  tool-ref: {args.tool_ref}")

    preds: list[str | None] = []
    golds: list[str] = []
    out_records: list[dict] = []
    unparsed = 0

    for i, rec in enumerate(records):
        # Deterministic option order keyed by record id.
        options = common.seeded_shuffle(
            [rec["correct_answer"], *rec["wrong_answers"]], rec["id"]
        )
        correct_letter = _LETTERS[options.index(rec["correct_answer"])]
        options_block = "\n".join(f"{_LETTERS[j]}. {opt}" for j, opt in enumerate(options))
        referent = common.tool_referent(rec["tool"], args.tool_ref)
        prompt = _PROMPT.format(referent=referent, question=rec["question"], options=options_block)

        if args.dry_run:
            raw = common.mock_letter(rec["id"])
        else:
            raw = model_client.complete(args.model, prompt, temperature=args.temperature)
        pred = common.parse_choice_letter(raw)
        if pred is None:
            unparsed += 1
        preds.append(pred)
        golds.append(correct_letter)
        out_records.append({"id": rec["id"], "gold": correct_letter, "pred": pred,
                            "correct": pred == correct_letter, "raw": raw})
        if (i + 1) % 50 == 0 or i + 1 == len(records):
            print(f"  [{i+1}/{len(records)}] running accuracy="
                  f"{metrics.accuracy(preds, golds):.3f}")

    acc = metrics.accuracy(preds, golds)
    correct = [1.0 if (p is not None and p == g) else 0.0 for p, g in zip(preds, golds)]
    ci_lo, ci_hi = metrics.bootstrap_ci(correct)
    common.write_jsonl(os.path.join(args.output, "mcq_predictions.jsonl"), out_records)
    card = f"""\
# MCQ Probing — Evaluation Results

| Field | Value |
|---|---|
| model | `{'<mock>' if args.dry_run else args.model}` |
| tool_ref | `{args.tool_ref}` |
| records | {len(records)} |
| **accuracy** | **{acc:.3f}** [{ci_lo:.3f}, {ci_hi:.3f}] |
| random baseline | 0.250 |
| unparseable responses | {unparsed} |

_accuracy 95% CI: percentile bootstrap (n_resamples=1000, seed=0)._
"""
    common.write_text(os.path.join(args.output, "mcq_results.md"), card)
    print(f"\nAccuracy: {acc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] (random 0.250)  unparseable: {unparsed}")
    print(f"Predictions -> {os.path.join(args.output, 'mcq_predictions.jsonl')}")
    print(f"Results card -> {os.path.join(args.output, 'mcq_results.md')}")


if __name__ == "__main__":
    main()
