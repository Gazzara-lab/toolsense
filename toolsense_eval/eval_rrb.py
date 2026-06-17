"""Evaluate a model on the ToolSense Realistic Retrieval Benchmark (RRB).

For each record the model is given the query and the shipped 14-tool candidate
pool (``analyzed_tools``, shuffled deterministically and numbered), and must rank
the candidates from most to least relevant. We score the ranking against the gold
tool set with Recall@k, hit-rate@k, MRR, and nDCG@k, broken down by complexity
tier (easy / medium / hard) and overall.

IMPORTANT — this is NOT a reproduction of the paper's Rc@50. The paper's Recall@50
is trie-constrained beam search over the full ~47k catalog produced by a *trained*
ToolGen retriever. Here the candidate set is the released 14-tool pool and the
model ranks it in-context; k therefore ranges over the pool size. See
REPRODUCTION.md.

Usage:
    python -m toolsense_eval.eval_rrb --data data/toolsense-realistic-retrieval/data.jsonl \\
        --output results/rrb --model claude-4.5-sonnet

    # Zero-cost end-to-end check:
    python -m toolsense_eval.eval_rrb --data data/toolsense-realistic-retrieval/data.jsonl \\
        --output results/rrb --dry-run
"""

from __future__ import annotations

import argparse
import os

from toolsense_eval import common, metrics, model_client

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "")
DEFAULT_KS = [1, 3, 5, 10]
TIERS = ["easy", "medium", "hard"]

_PROMPT = """\
You are a tool-retrieval system. Given a user query and a numbered list of candidate \
tools, rank ALL candidates from MOST to LEAST relevant for fulfilling the query.

User query: {query}

Candidate tools:
{candidates}

Return ONLY a comma-separated list of the candidate numbers, ordered from most to \
least relevant (for example: 3, 1, 7, ...). Include every number from 1 to {n} \
exactly once. Do not explain."""


def _candidate_block(pool: list[dict]) -> str:
    lines = []
    for j, tool in enumerate(pool, start=1):
        name = tool.get("tool_name", "")
        desc = tool.get("tool_description", "") or tool.get("api_description", "")
        lines.append(f"{j}. {name} — {desc}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="RRB benchmark JSONL")
    parser.add_argument("--output", required=True, help="Output directory for predictions + results card")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LiteLLM model string (default: {DEFAULT_MODEL!r})")
    parser.add_argument("--tier", choices=TIERS, default=None, help="Restrict to one complexity tier")
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_KS,
                        help=f"Recall@k cutoffs (default: {DEFAULT_KS})")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Max records to evaluate (default: all)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--dry-run", "--mock", dest="dry_run", action="store_true",
                        help="Score against a deterministic stub responder with no API calls")
    args = parser.parse_args()

    if any(k < 1 for k in args.k):
        parser.error("--k values must be >= 1")

    if not args.dry_run:
        model_client.load_env()
        if not args.model:
            parser.error("--model is required unless --dry-run is set (or set DEFAULT_MODEL)")

    records = common.load_jsonl(args.data)
    if args.tier:
        records = [r for r in records if r.get("complexity") == args.tier]
    if args.num_samples is not None:
        records = records[: args.num_samples]
    print(f"Loaded {len(records)} RRB records from {args.data}")
    print(f"Model: {'<mock>' if args.dry_run else args.model}  tiers: {args.tier or 'all'}")

    out_records: list[dict] = []
    # per_tier[tier] = list of per-record metric dicts
    per_tier: dict[str, list[dict]] = {t: [] for t in TIERS}
    unparsed = 0
    pool_sizes: set[int] = set()

    for i, rec in enumerate(records):
        pool = common.seeded_shuffle(rec["analyzed_tools"], rec["sample_id"])
        n = len(pool)
        pool_sizes.add(n)
        gold = set(common.gold_tool_names(rec))
        prompt = _PROMPT.format(query=rec["query"], candidates=_candidate_block(pool), n=n)

        if args.dry_run:
            raw = common.mock_ranking(rec["sample_id"], n)
        else:
            raw = model_client.complete(args.model, prompt, temperature=args.temperature)
        ranked_idx, n_parsed = common.parse_ranking(raw, n)
        # A response with no in-range index is a non-answer: score it as an empty
        # ranking (all metrics 0) rather than crediting the auto-completed order.
        unparseable = n_parsed == 0
        if unparseable:
            unparsed += 1
            ranked_names: list[str] = []
        else:
            ranked_names = [pool[j - 1]["tool_name"] for j in ranked_idx]

        row = {"recall": {}, "hit": {}}
        for k in args.k:
            row["recall"][k] = metrics.recall_at_k(ranked_names, gold, k)
            row["hit"][k] = metrics.hit_at_k(ranked_names, gold, k)
        row["mrr"] = metrics.reciprocal_rank(ranked_names, gold)
        row["ndcg"] = {k: metrics.ndcg_at_k(ranked_names, gold, k) for k in args.k}

        tier = rec.get("complexity", "unknown")
        if tier in per_tier:
            per_tier[tier].append(row)
        out_records.append({
            "sample_id": rec["sample_id"], "complexity": tier, "gold_count": len(gold),
            "ranked_tool_names": ranked_names, "unparseable": unparseable,
            "n_parsed": n_parsed, "raw": raw,
            "recall": row["recall"], "hit": row["hit"], "mrr": row["mrr"], "ndcg": row["ndcg"],
        })
        if (i + 1) % 50 == 0 or i + 1 == len(records):
            print(f"  [{i+1}/{len(records)}] processed")

    def agg(rows: list[dict]) -> dict:
        if not rows:
            return {}
        out = {"n": len(rows), "recall": {}, "hit": {}, "ndcg": {}}
        for k in args.k:
            out["recall"][k] = metrics.mean([r["recall"][k] for r in rows])
            out["hit"][k] = metrics.mean([r["hit"][k] for r in rows])
            out["ndcg"][k] = metrics.mean([r["ndcg"][k] for r in rows])
        out["mrr"] = metrics.mean([r["mrr"] for r in rows])
        return out

    all_rows = [r for rows in per_tier.values() for r in rows]
    overall = agg(all_rows)
    # 95% bootstrap CIs on the headline overall metrics (per-tier rows keep their n).
    overall_ci = {k: metrics.bootstrap_ci([r["recall"][k] for r in all_rows]) for k in args.k}
    overall_ci_mrr = metrics.bootstrap_ci([r["mrr"] for r in all_rows]) if all_rows else (0.0, 0.0)

    common.write_jsonl(os.path.join(args.output, "rrb_predictions.jsonl"), out_records)

    # Pool size is computed from the data (not assumed constant): a single value
    # if every record shares it, else a min-max range.
    if not pool_sizes:
        pool_size_str = "0"
    elif len(pool_sizes) == 1:
        pool_size_str = str(next(iter(pool_sizes)))
    else:
        pool_size_str = f"{min(pool_sizes)}-{max(pool_sizes)}"

    # Build results card.
    header = "| tier | n | " + " | ".join(f"R@{k}" for k in args.k) + " | MRR | " + \
             " | ".join(f"nDCG@{k}" for k in args.k) + " |"
    sep = "|" + "---|" * (2 + 2 * len(args.k) + 1)

    def row_md(name: str, a: dict) -> str:
        if not a:
            return f"| {name} | 0 | " + " | ".join("—" for _ in range(2 * len(args.k) + 1)) + " |"
        cells = [f"{a['recall'][k]:.3f}" for k in args.k] + [f"{a['mrr']:.3f}"] + \
                [f"{a['ndcg'][k]:.3f}" for k in args.k]
        return f"| {name} | {a['n']} | " + " | ".join(cells) + " |"

    lines = [
        "# Realistic Retrieval (RRB) — Evaluation Results",
        "",
        f"Model: `{'<mock>' if args.dry_run else args.model}`  ·  candidate pool size: {pool_size_str}  ·  "
        f"records: {len(records)}  ·  unparseable responses: {unparsed}",
        "",
        f"Recall@k / hit-rate@k / MRR / nDCG@k over the shipped {pool_size_str}-tool candidate pool. "
        "Not comparable to the paper's trained-ToolGen Rc@50 over the full catalog.",
        "",
        header, sep,
        row_md("overall", overall),
    ]
    for t in TIERS:
        if per_tier[t]:
            lines.append(row_md(t, agg(per_tier[t])))
    if all_rows:
        ci_parts = [f"R@{k} [{overall_ci[k][0]:.3f}, {overall_ci[k][1]:.3f}]" for k in args.k]
        ci_parts.append(f"MRR [{overall_ci_mrr[0]:.3f}, {overall_ci_mrr[1]:.3f}]")
        lines += ["", "Overall 95% CIs (percentile bootstrap, n_resamples=1000, seed=0): "
                  + ", ".join(ci_parts)]
    # hit-rate table
    lines += ["", "Hit-rate@k (fraction of queries with ≥1 gold tool in top-k):", "",
              "| tier | n | " + " | ".join(f"hit@{k}" for k in args.k) + " |",
              "|" + "---|" * (2 + len(args.k))]

    def hit_row(name: str, a: dict) -> str:
        if not a:
            return f"| {name} | 0 | " + " | ".join("—" for _ in args.k) + " |"
        return f"| {name} | {a['n']} | " + " | ".join(f"{a['hit'][k]:.3f}" for k in args.k) + " |"

    lines.append(hit_row("overall", overall))
    for t in TIERS:
        if per_tier[t]:
            lines.append(hit_row(t, agg(per_tier[t])))

    common.write_text(os.path.join(args.output, "rrb_results.md"), "\n".join(lines) + "\n")
    print("\nOverall: " + ", ".join(f"R@{k}={overall['recall'][k]:.3f}" for k in args.k)
          + f", MRR={overall['mrr']:.3f}  unparseable: {unparsed}")
    print(f"Predictions -> {os.path.join(args.output, 'rrb_predictions.jsonl')}")
    print(f"Results card -> {os.path.join(args.output, 'rrb_results.md')}")


if __name__ == "__main__":
    main()
