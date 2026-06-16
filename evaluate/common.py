"""Shared utilities for the ToolSense evaluation harness.

Pure standard library so the eval path and mock mode have no heavy dependencies.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path


# ----------------------------------------------------------------------------- IO
def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts (blank lines skipped)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def write_text(path: str | Path, text: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


# ----------------------------------------------------------- determinism helpers
def _seed_int(seed_str: str) -> int:
    return int(hashlib.md5(seed_str.encode("utf-8")).hexdigest(), 16)


def seeded_rng(seed_str: str) -> random.Random:
    """A deterministic RNG keyed by an arbitrary string (e.g. a record id)."""
    return random.Random(_seed_int(seed_str))


def seeded_shuffle(seq: list, seed_str: str) -> list:
    """Return a deterministically shuffled copy of ``seq`` keyed by ``seed_str``."""
    out = list(seq)
    seeded_rng(seed_str).shuffle(out)
    return out


# -------------------------------------------------------------- tool referent text
def tool_referent(tool: dict, mode: str) -> str:
    """Render the text that grounds the phrase 'this tool' in a probe question.

    The paper hides the tool name and shows a trained virtual token. A generic
    API model has no such token, so we substitute a textual referent and let the
    caller choose how much to reveal:

      name        tool name + title only (in-context name-identity probe, default;
                  for descriptive names this is largely surface string-matching,
                  not parametric knowledge — see REPRODUCTION.md)
      description name + title + description (a reading-comprehension upper bound)
      none        nothing identifying (a blind lower bound)
    """
    name = tool.get("tool_name", "")
    title = tool.get("tool_title", "") or tool.get("api_title", "")
    if mode == "none":
        return "(The tool's identity is hidden.)"
    if mode == "description":
        desc = tool.get("tool_description", "") or tool.get("api_description", "")
        return f'Tool name: {name}\nTool title: {title}\nTool description: {desc}'
    # default: name
    return f"Tool name: {name}\nTool title: {title}"


# --------------------------------------------------------------------- parsing
# Real models often ignore "answer with one word/letter" and explain. The parsers
# below are tuned (see the regression tests) so that a model that reasons before
# committing is read by its *concluding* answer, not the first token it happens to
# mention — while genuinely ambiguous or non-committal output stays unparseable
# (which the harness counts) rather than being guessed wrong.
_YES = re.compile(r"\b(yes)\b", re.IGNORECASE)
_NO = re.compile(r"\b(no)\b", re.IGNORECASE)

# An explicitly *declared* yes/no: a cue word, then within a short same-line window
# (lowercase words allowed, never crossing a newline), a yes/no token; or a yes/no
# standing as the final sentence. The LAST declaration wins, so an explanation that
# mentions the opposite word first ("... the older version said yes, so no") is read
# by its conclusion.
_DECLARED_YN = re.compile(
    r"(?i)(?:"
    r"\b(?:answer|response|verdict|conclusion)\b.{0,15}?\b(yes|no)\b"
    r"|\b(?:so|therefore|thus|hence|overall|finally)\b.{0,8}?\b(yes|no)\b"
    r"|\bfinal answer\b.{0,15}?\b(yes|no)\b"
    r"|\b(?:say|leans? towards?|leaning towards?|go(?:ing)? with)\b.{0,10}?\b(yes|no)\b"
    r"|[.!?\n]\s*(yes|no)\b\W*$"
    r")"
)
# Last-resort synonyms, matched only at the very start of the response and only when
# no clean yes/no token exists (small models that answer "Nope"/"Yeah"/"Y"/"True").
_SYN_YES = re.compile(r"^\W*(?:y|yeah|yep|yup|sure|affirmative|true)\b", re.IGNORECASE)
_SYN_NO = re.compile(r"^\W*(?:n|nope|nah|negative|false)\b", re.IGNORECASE)

# MCQ answer extraction. Letters are matched UPPERCASE-only (the prompt presents
# options as A/B/C/D and asks for a letter), so lowercase prose — the article "a",
# or words like "be"/"cat"/"would" — is never read as a choice.
# A decision cue, then (lowercase words allowed, but no other uppercase letter or
# newline in between) a single choice letter. The LAST such match wins, so a model
# that restates "options A, B, C, D ... I pick D" is read as D, not A.
_CHOICE_DECISION = re.compile(
    r"(?i)\b(?:final answer|answer|correct(?:\s+\w+)?|pick|choose|chose|select|"
    r"go with|going with|opt for|the (?:answer|option|choice)|my (?:answer|choice))\b"
    r"[^A-Za-z\n]{0,12}\(?([A-D])\)?(?![A-Za-z])"
)
_CHOICE_DELIM = re.compile(r"(?<![A-Za-z])([A-D])\s*[).:,]")
_CHOICE_PAREN = re.compile(r"\(([A-D])\)")
_CHOICE_STANDALONE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")
_ARTICLE_A = re.compile(r"\s+[a-z]")
# An enumeration head: the matched letter is immediately followed by more list
# letters ("A, B" / "A and B"), i.e. a restated option list, not the chosen answer.
_ENUM_AFTER = re.compile(r"^[^A-Za-z\n]{0,4}(?:and\s+|or\s+)?[A-D]\b", re.IGNORECASE)

# Ranking index token: a run of digits not part of a decimal/dotted token
# (so "1.2" or "v3.4" do not leak stray indices into the ranking).
_INT_TOKEN = re.compile(r"(?<![\d.])\d+(?![\d.])")
# A leading list enumerator ("1. " / "2) ") at the start of a line.
_LIST_MARKER = re.compile(r"^\s*(\d+)[.)]\s+\S")


def parse_yes_no(text: str) -> str | None:
    """Return 'Yes'/'No' from a model response, or None if unparseable.

    An explicitly declared answer (the last one, if a model explains) wins; else the
    earliest clean yes/no token; else a conservative leading synonym.
    """
    decl = list(_DECLARED_YN.finditer(text))
    if decl:
        token = next(g for g in decl[-1].groups() if g)
        return "Yes" if token.lower() == "yes" else "No"
    y = _YES.search(text)
    n = _NO.search(text)
    if y and (not n or y.start() < n.start()):
        return "Yes"
    if n and (not y or n.start() < y.start()):
        return "No"
    s = text.strip()
    if _SYN_YES.match(s):
        return "Yes"
    if _SYN_NO.match(s):
        return "No"
    return None


def parse_choice_letter(text: str) -> str | None:
    """Return the chosen MCQ letter (A/B/C/D), or None if unparseable.

    Staged for precision: a bare single letter; then the last explicit decision
    ("... the answer is C", "I pick D") skipping restated option lists; then the last
    delimited letter ("B." / "A)" / "(C)"); then any standalone uppercase A-D —
    skipping a lone "A" that is the English article (followed by a space + lowercase
    word). Lowercase letters embedded in prose are intentionally not read as choices.
    """
    s = text.strip()
    if len(s) == 1 and s.upper() in "ABCD":
        return s.upper()
    decisions = [m for m in _CHOICE_DECISION.finditer(text)
                 if not _ENUM_AFTER.match(text[m.end():])]
    if decisions:
        return decisions[-1].group(1)
    delims = [m for m in _CHOICE_DELIM.finditer(text)
              if not _ENUM_AFTER.match(text[m.end():])]
    if delims:
        return delims[-1].group(1)
    m = _CHOICE_PAREN.search(text)
    if m:
        return m.group(1)
    for m in _CHOICE_STANDALONE.finditer(text):
        letter = m.group(1)
        if letter == "A" and _ARTICLE_A.match(text[m.end():]):
            continue
        return letter
    return None


def _strip_rank_markers(text: str) -> str:
    """Drop leading "N." / "N)" list enumerators when they form a 1..k sequence.

    A ranked list like ``1. Tool 7`` / ``2. Tool 14`` carries the rank position as a
    leading number; stripping it leaves the real tool ids to be parsed. Only fires on
    a clean sequential 1,2,3,..., so it never removes genuine answer digits.
    """
    lines = text.splitlines()
    markers = [int(m.group(1)) if (m := _LIST_MARKER.match(ln)) else None for ln in lines]
    seq = [m for m in markers if m is not None]
    if len(seq) >= 3 and seq == list(range(1, len(seq) + 1)):
        return "\n".join(re.sub(r"^\s*\d+[.)]\s+", "", ln, count=1) for ln in lines)
    return text


def parse_ranking(text: str, n: int) -> tuple[list[int], int]:
    """Extract a 1-based ranking of candidate indices in [1, n].

    Returns ``(ranking, n_parsed)`` where ``ranking`` is a COMPLETE permutation
    of 1..n (the model's in-range indices first, in order, then any missing
    indices appended) and ``n_parsed`` is how many valid distinct indices the
    model actually supplied. ``n_parsed == 0`` means the response was unparseable
    (no in-range index) and the caller should treat it as a non-answer rather than
    crediting the auto-completed identity order.
    """
    text = _strip_rank_markers(text)
    seen: list[int] = []
    for tok in _INT_TOKEN.findall(text):
        v = int(tok)
        if 1 <= v <= n and v not in seen:
            seen.append(v)
    n_parsed = len(seen)
    for v in range(1, n + 1):
        if v not in seen:
            seen.append(v)
    return seen, n_parsed


# -------------------------------------------------------- deterministic mock model
def mock_yes_no(key: str) -> str:
    return "Yes" if _seed_int(key) % 2 == 0 else "No"


def mock_letter(key: str) -> str:
    return "ABCD"[_seed_int(key) % 4]


def mock_ranking(key: str, n: int) -> str:
    order = seeded_shuffle(list(range(1, n + 1)), key)
    return ", ".join(str(i) for i in order)


# --------------------------------------------------------------------- gold sets
def gold_tool_names(record: dict) -> list[str]:
    """Normalize the polymorphic RRB 'tool' field to a list of tool names.

    Easy records carry a single dict; medium/hard carry a list of dicts.
    """
    t = record["tool"]
    if isinstance(t, list):
        return [x["tool_name"] for x in t]
    return [t["tool_name"]]
