"""Evaluate a model on the ToolSense QA probing benchmark (yes/no).

For each record the model is shown a textual referent for "this tool" (by default
its name + title) and the yes/no question, and must answer "Yes" or "No". We
report exact-match accuracy against the 50% random baseline.

Usage:
    python -m evaluate.eval_qa \\
        --data data/toolsense-qa/data.jsonl \\
        --output results/qa \\
        --model claude-4.5-sonnet

    # Zero-cost end-to-end check (deterministic stub, no API calls):
    python -m evaluate.eval_qa --data data/toolsense-qa/data.jsonl \\
        --output results/qa --dry-run

Scope: this is an inference-only probe of what a model knows about a tool given
its public identifier; it is not a reproduction of the paper's trained-token Rc@k.
"""

from __future__ import annotations

import argparse
import os

from evaluate import common, metrics, model_client

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "")

_PROMPT = """\
You are evaluating tool knowledge. Consider the following tool.

{referent}

Answer the following yes/no question about this tool.
Question: {question}

Respond with exactly one word: "Yes" or "No". Do not explain."""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True,
                        help="QA benchmark JSONL (e.g. data/toolsense-qa/data.jsonl)")
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
    print(f"Loaded {len(records)} QA records from {args.data}")
    print(f"Model: {'<mock>' if args.dry_run else args.model}  tool-ref: {args.tool_ref}")

    preds: list[str | None] = []
    golds: list[str] = []
    out_records: list[dict] = []
    unparsed = 0

    for i, rec in enumerate(records):
        referent = common.tool_referent(rec["tool"], args.tool_ref)
        prompt = _PROMPT.format(referent=referent, question=rec["question"])
        if args.dry_run:
            raw = common.mock_yes_no(rec["id"])
        else:
            raw = model_client.complete(args.model, prompt, temperature=args.temperature)
        pred = common.parse_yes_no(raw)
        if pred is None:
            unparsed += 1
        gold = rec["answer"]
        preds.append(pred)
        golds.append(gold)
        out_records.append({"id": rec["id"], "gold": gold, "pred": pred,
                            "correct": pred == gold, "raw": raw[:200]})
        if (i + 1) % 50 == 0 or i + 1 == len(records):
            print(f"  [{i+1}/{len(records)}] running accuracy="
                  f"{metrics.accuracy(preds, golds):.3f}")

    acc = metrics.accuracy(preds, golds)
    correct = [1.0 if (p is not None and p == g) else 0.0 for p, g in zip(preds, golds)]
    ci_lo, ci_hi = metrics.bootstrap_ci(correct)
    yes_frac = sum(1 for g in golds if g == "Yes") / len(golds) if golds else 0.0

    common.write_jsonl(os.path.join(args.output, "qa_predictions.jsonl"), out_records)
    card = f"""\
# QA Probing — Evaluation Results

| Field | Value |
|---|---|
| model | `{'<mock>' if args.dry_run else args.model}` |
| tool_ref | `{args.tool_ref}` |
| records | {len(records)} |
| **accuracy** | **{acc:.3f}** [{ci_lo:.3f}, {ci_hi:.3f}] |
| random baseline | 0.500 |
| unparseable responses | {unparsed} |
| gold Yes fraction | {yes_frac:.3f} |

_accuracy 95% CI: percentile bootstrap (n_resamples=1000, seed=0)._
"""
    common.write_text(os.path.join(args.output, "qa_results.md"), card)
    print(f"\nAccuracy: {acc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] (random 0.500)  unparseable: {unparsed}")
    print(f"Predictions -> {os.path.join(args.output, 'qa_predictions.jsonl')}")
    print(f"Results card -> {os.path.join(args.output, 'qa_results.md')}")


if __name__ == "__main__":
    main()
