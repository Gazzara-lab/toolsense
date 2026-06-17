"""Model-agnostic evaluation harness for the ToolSense diagnostic benchmarks.

The ToolSense repo ships the three diagnostic *datasets* (QA, MCQ, Realistic
Retrieval) and generation code, but no turnkey scorer to run a model against them
and report the headline metrics. This package fills that gap with three CLI
entry points:

    eval-qa    QA probing  — yes/no accuracy vs the 50% random baseline
    eval-mcq   MCQ probing — 4-way accuracy vs the 25% random baseline
    eval-rrb   Realistic Retrieval — Recall@k / hit-rate / MRR / nDCG over the
               shipped 14-tool candidate pool, broken down by complexity tier

Model calls mirror the generators' pattern: when ``LITELLM_BASE_URL`` is set they
route through an OpenAI-compatible proxy (the ``openai`` client), otherwise they
call ``litellm.completion`` directly. Either way any OpenAI-compatible model,
provider, or proxy works. Every command also supports a
``--dry-run`` mock mode that scores against a deterministic stub responder with
**zero** paid API calls — useful for CI and for verifying the harness end-to-end.

Scope note: this harness measures an *inference-only*, in-context quantity. It is
NOT a reproduction of the paper's trained-ToolGen Rc@50 numbers (those use a
fine-tuned parametric retriever with trie-constrained beam search over the full
~47k catalog). See REPRODUCTION.md for the precise distinction.
"""

__all__ = ["metrics", "model_client", "common"]
