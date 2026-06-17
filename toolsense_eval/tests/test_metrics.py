"""Known-answer tests for the evaluation metrics and parsers (offline, no API)."""

import math

from toolsense_eval import common, metrics


# --------------------------------------------------------------------- metrics
def test_recall_at_k_basic():
    ranked = ["a", "b", "c", "d"]
    gold = {"a", "c"}
    assert metrics.recall_at_k(ranked, gold, 1) == 0.5      # only "a" in top-1
    assert metrics.recall_at_k(ranked, gold, 3) == 1.0      # "a" and "c" in top-3
    assert metrics.recall_at_k(ranked, gold, 4) == 1.0


def test_recall_capped_by_k():
    ranked = ["a", "b", "c", "d"]
    gold = {"a", "b", "c"}
    # top-1 can contain at most 1 of 3 gold -> 1/3
    assert abs(metrics.recall_at_k(ranked, gold, 1) - 1 / 3) < 1e-9


def test_recall_empty_gold():
    assert metrics.recall_at_k(["a"], set(), 5) == 0.0


def test_hit_at_k():
    ranked = ["x", "a", "y"]
    gold = {"a"}
    assert metrics.hit_at_k(ranked, gold, 1) == 0.0
    assert metrics.hit_at_k(ranked, gold, 2) == 1.0


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert metrics.reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert metrics.reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_ndcg_perfect_is_one():
    ranked = ["a", "b", "c", "d"]
    gold = {"a", "b"}
    assert abs(metrics.ndcg_at_k(ranked, gold, 4) - 1.0) < 1e-9


def test_ndcg_known_value():
    # gold at ranks 1 and 3: DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # ideal (gold at ranks 1,2): IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    ranked = ["a", "x", "b", "y"]
    gold = {"a", "b"}
    expected = (1 + 0.5) / (1 + 1 / math.log2(3))
    assert abs(metrics.ndcg_at_k(ranked, gold, 4) - expected) < 1e-9


def test_accuracy_with_none():
    assert metrics.accuracy(["Yes", None, "No"], ["Yes", "No", "No"]) == 2 / 3


# --------------------------------------------------------------------- parsers
def test_parse_yes_no():
    assert common.parse_yes_no("Yes.") == "Yes"
    assert common.parse_yes_no("The answer is No") == "No"
    assert common.parse_yes_no("Yes, definitely not No") == "Yes"   # first wins
    assert common.parse_yes_no("maybe") is None


def test_parse_choice_letter():
    assert common.parse_choice_letter("C") == "C"
    assert common.parse_choice_letter("c") == "C"
    assert common.parse_choice_letter("The answer is B.") == "B"
    assert common.parse_choice_letter("(D)") == "D"
    assert common.parse_choice_letter("B)") == "B"
    assert common.parse_choice_letter("none here") is None


def test_parse_choice_letter_normalizes_case():
    # A lowercase decision letter captured by the case-insensitive decision regex
    # must be normalized to the uppercase A-D gold labels, else a correctly-parsed
    # answer is scored wrong.
    assert common.parse_choice_letter("Final answer: c") == "C"
    assert common.parse_choice_letter("I would pick d") == "D"
    assert common.parse_choice_letter("I choose b") == "B"
    assert common.parse_choice_letter("answer: a") == "A"


def test_parse_choice_letter_ignores_article_a():
    # The English article "a"/"A" must not be read as choice A.
    assert common.parse_choice_letter("A reasonable choice would be option B.") == "B"
    assert common.parse_choice_letter("Based on the description, a suitable answer is C.") == "C"
    assert common.parse_choice_letter("A tool like this maps to D.") == "D"
    assert common.parse_choice_letter("a tool") is None
    assert common.parse_choice_letter("It is a cat") is None
    # A genuine bare A still works.
    assert common.parse_choice_letter("A") == "A"
    assert common.parse_choice_letter("A.") == "A"


def test_parse_ranking_completes_and_dedups():
    # n=4, model lists 3,1,3 -> keep [3,1] (2 parsed), append missing 2,4
    ranking, n_parsed = common.parse_ranking("3, 1, 3", 4)
    assert ranking == [3, 1, 2, 4]
    assert n_parsed == 2
    # out-of-range ignored
    ranking, n_parsed = common.parse_ranking("9, 2", 4)
    assert ranking == [2, 1, 3, 4]
    assert n_parsed == 1


def test_parse_ranking_empty_is_flagged():
    ranking, n_parsed = common.parse_ranking("I cannot rank these", 5)
    assert ranking == [1, 2, 3, 4, 5]  # still a complete permutation
    assert n_parsed == 0               # but flagged as a non-answer


def test_parse_ranking_ignores_decimals():
    # "1.2" must not inject indices 1 and 2; only the clean integer 3 is parsed.
    ranking, n_parsed = common.parse_ranking("1.2, 3", 4)
    assert n_parsed == 1
    assert ranking[0] == 3


def test_gold_tool_names_polymorphic():
    easy = {"tool": {"tool_name": "A"}}
    multi = {"tool": [{"tool_name": "A"}, {"tool_name": "B"}]}
    assert common.gold_tool_names(easy) == ["A"]
    assert common.gold_tool_names(multi) == ["A", "B"]


def test_determinism():
    a = common.seeded_shuffle([1, 2, 3, 4, 5], "key")
    b = common.seeded_shuffle([1, 2, 3, 4, 5], "key")
    assert a == b
    assert common.mock_letter("x") == common.mock_letter("x")


# ----------------------------------------------- robustness to real model output
def test_parse_yes_no_declared_answer_wins_over_earlier_mention():
    # A model that explains concludes with its answer; an opposite word mentioned
    # earlier in the reasoning must not flip the result.
    assert common.parse_yes_no(
        "While the older version said yes, the current release dropped it, so no.") == "No"
    assert common.parse_yes_no(
        "The documentation says no manual setup is required, so yes, it works.") == "Yes"
    assert common.parse_yes_no(
        "The tool has yes/no toggles, but the answer is no.") == "No"


def test_parse_yes_no_compliant_unaffected():
    # A leading clean answer with an incidental opposite word later stays correct.
    assert common.parse_yes_no("Yes, although there is no evidence of issues.") == "Yes"
    assert common.parse_yes_no("No. This tool does not support streaming.") == "No"


def test_parse_yes_no_synonyms_and_nonanswer():
    assert common.parse_yes_no("Nope, that was deprecated.") == "No"
    assert common.parse_yes_no("Yeah, it supports webhooks.") == "Yes"
    assert common.parse_yes_no("Y") == "Yes"
    assert common.parse_yes_no("N") == "No"
    assert common.parse_yes_no("True.") == "Yes"
    # a genuine non-answer is still unparseable (counted, not guessed)
    assert common.parse_yes_no("It depends entirely on your configuration.") is None


def test_parse_choice_letter_restated_options_then_answer():
    # The decision regex must not grab the first letter of a restated option list.
    assert common.parse_choice_letter(
        "The options are A, B, C, and D. After reviewing them, I pick D.") == "D"
    assert common.parse_choice_letter(
        "Option A is wrong because it overstates the limit. The answer is C.") == "C"
    assert common.parse_choice_letter(
        "Choice A says X. Choice B says Y. I'll go with D.") == "D"
    assert common.parse_choice_letter(
        "Among A, B, C, D, the one that matches the docs is D.") == "D"
    assert common.parse_choice_letter(
        "Options A and D are incorrect, B is a distractor. The correct choice is C.") == "C"


def test_parse_ranking_strips_rank_list_markers():
    # "1. Tool 7 / 2. Tool 14 / ..." — the leading 1,2,... are rank positions, the
    # real tool ids follow; a compliant single-line comma list is left untouched.
    ranking, n_parsed = common.parse_ranking("1. Tool 7\n2. Tool 14\n3. Tool 2", 14)
    assert ranking[:3] == [7, 14, 2]
    assert n_parsed == 3
    ranking, n_parsed = common.parse_ranking("3, 1, 2, 4", 4)
    assert ranking == [3, 1, 2, 4]
    assert n_parsed == 4


# ------------------------------------------------------------- bootstrap CIs
def test_bootstrap_ci_all_correct_collapses_to_ceiling():
    lo, hi = metrics.bootstrap_ci([1.0] * 100)
    assert lo == 1.0 and hi == 1.0


def test_bootstrap_ci_is_deterministic_for_a_seed():
    scores = ([1.0] * 60) + ([0.0] * 40)
    assert metrics.bootstrap_ci(scores, seed=0) == metrics.bootstrap_ci(scores, seed=0)


def test_bootstrap_ci_brackets_estimate_and_narrows_with_n():
    lo, hi = metrics.bootstrap_ci(([1.0] * 60) + ([0.0] * 40), seed=0)   # mean 0.60, n=100
    assert lo < 0.60 < hi
    lo2, hi2 = metrics.bootstrap_ci(([1.0] * 600) + ([0.0] * 400), seed=0)  # mean 0.60, n=1000
    assert (hi2 - lo2) < (hi - lo)


def test_bootstrap_ci_half_width_matches_wald_ballpark():
    scores = ([1.0] * 60) + ([0.0] * 40)  # p=0.6, n=100
    lo, hi = metrics.bootstrap_ci(scores, n_resamples=4000, seed=0)
    wald = 1.96 * math.sqrt(0.6 * 0.4 / 100)
    assert abs((hi - lo) / 2 - wald) < 0.02


def test_bootstrap_diff_ci_identical_conditions_bracket_zero():
    v = ([1.0, 0.0] * 50)
    lo, hi = metrics.bootstrap_diff_ci(v, v, seed=0)
    assert lo <= 0.0 <= hi


def test_bootstrap_diff_ci_clear_lift_is_above_zero():
    a = ([1.0] * 90) + ([0.0] * 10)   # 0.90
    b = ([1.0] * 50) + ([0.0] * 50)   # 0.50
    lo, hi = metrics.bootstrap_diff_ci(a, b, seed=0)
    assert lo > 0.0
