"""Tests for xai_health_bench.text."""
from xai_health_bench.text import (
    bm25_scores,
    expand_question,
    extract_citations,
    numbers_supported,
    overlap_ratio,
    phrase_present,
    question_coverage,
    split_sentences,
    stem,
    tokenize,
)


def test_tokenize_lowercases_and_keeps_numbers():
    assert tokenize("HbA1c below 7.0%") == ["hba1c", "below", "7.0"]


def test_stem_e_strip_makes_forms_match():
    # The e-strip rule: "cause"/"causes", "reduce"/"reduced" must stem alike.
    assert stem("cause") == stem("causes") == "caus"
    assert stem("reduce") == stem("reduced") == "reduc"
    assert stem("take") == stem("taking") == "tak"
    assert stem("make") == stem("making") == "mak"
    assert stem("dose") == stem("doses") == "dos"
    assert stem("screening") == stem("screened") == "screen"


def test_stem_weekly_to_week():
    assert stem("weekly") == "week"


def test_overlap_ratio_full_and_zero():
    assert overlap_ratio("metformin dose reviewed", "metformin dose reviewed") == 1.0
    assert overlap_ratio("metformin dose", "insulin storage") == 0.0


def test_numbers_supported_requires_verbatim_numbers():
    assert numbers_supported("HbA1c below 7.0%", "aim for HbA1c below 7.0%")
    assert not numbers_supported("HbA1c below 8.0%", "aim for HbA1c below 7.0%")


def test_phrase_present_needs_all_tokens_in_one_text():
    assert phrase_present("below 7.0%", "aim for an HbA1c below 7.0%")
    assert not phrase_present("below 7.0%", "aim for an HbA1c below 8.0%")
    assert not phrase_present("lung deposition", "spacer improves delivery")


def test_extract_citations():
    assert extract_citations("Claim one [S1]. Claim two [S2].") == [1, 2]
    assert extract_citations("No citations here.") == []


def test_split_sentences_skips_short_fragments():
    parts = split_sentences(
        "First sentence here is long enough. Second sentence here is long too.")
    assert len(parts) == 2
    assert split_sentences("Too short.") == []


def test_bm25_ranks_matching_doc_first():
    docs = ["metformin and kidney disease", "insulin storage in the fridge",
            "hand hygiene in hospitals"]
    scores = bm25_scores("metformin kidney", docs)
    assert scores[0] > scores[1] and scores[0] > scores[2]


def test_bm25_empty_docs():
    assert bm25_scores("anything", []) == []


def test_question_coverage():
    assert question_coverage("Metformin in kidney disease", "metformin kidney") > 0.5
    assert question_coverage("Insulin storage", "metformin kidney") == 0.0


def test_expand_question_adds_synonyms():
    expanded = expand_question("How much weekly exercise is recommended?")
    assert "activity" in expanded
    assert "week" in expanded
