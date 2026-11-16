"""Small text helpers: tokenisation, stemming, overlap, BM25.

Adapted from the conventions of ``health_agent.text_utils`` (the
healthcare-research-agent repo): same stopword list, same crude stemmer,
same overlap definition — so the two projects' numbers are comparable.
Everything is standard-library only and deterministic.
"""
from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = frozenset(
    """a an the and or of for to in on at by with from as is are was were be been being
    do does did what which who whom when where why how can could should would may might
    i you he she it we they this that these those there here about into than then so such
    if but not no yes have has had will shall per""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_CITATION_RE = re.compile(r"\[S(\d+)\]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def stem(token: str) -> str:
    """Crude suffix stripping so 'screening'/'screened'/'screens' match.

    Not a Porter stemmer — just enough normalisation for lexical matching
    in a small corpus, and easy to reason about.

    The final rule matters most: after stripping a suffix, a trailing
    "e" is removed too ("cause" -> "caus", "reduce" -> "reduc",
    "take" -> "tak"). Without it the stemmer is inconsistent —
    "causes" stemmed to "caus" while "cause" stayed "cause", so the
    same word in the question and the evidence never matched. (Caught
    by the bench's own validation run on xh-017.)
    """
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 5 and token.endswith("ly"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    if len(token) > 2 and token.endswith("e"):
        token = token[:-1]
    return token


# Lexical bridge for concept pairs this corpus uses in different words
# (question says "weekly exercise", evidence says "minutes of moderate
# activity per week"). A small, explicit, documented map — not a hidden
# semantic model. Applied to the question before BM25 and coverage.
SYNONYMS: dict[str, list[str]] = {
    "exercise": ["activity"],
    "delivery": ["deposition"],
    "improve": ["increase"],
    "stop": ["discontinue", "halt"],
    "safe": ["safety", "avoided"],
    "prevent": ["reduce"],
    "dose": ["dosage"],
    "amount": ["dose"],
}


def expand_question(question: str) -> str:
    """Question text plus synonym words for terms in SYNONYMS."""
    tokens = tokenize(question)
    extra: list[str] = []
    for tok in tokens:
        extra.extend(SYNONYMS.get(tok, []))
    return question + (" " + " ".join(extra) if extra else "")


def stemmed_tokens(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text)]


def content_tokens(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def overlap_ratio(claim: str, evidence: str) -> float:
    """Fraction of distinct claim content tokens present in evidence."""
    claim_tokens = set(content_tokens(claim))
    if not claim_tokens:
        return 0.0
    evidence_tokens = set(stemmed_tokens(evidence))
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def numbers_in(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def numbers_supported(claim: str, evidence: str) -> bool:
    """Every number in the claim must appear verbatim in the evidence."""
    return all(n in evidence for n in numbers_in(claim))


def phrase_present(phrase: str, text: str) -> bool:
    """True when every content token of ``phrase`` appears in ``text``.

    Lexical proxy for "the answer states this fact". Documented
    limitation: it can miss a correct paraphrase and can be fooled by a
    sentence that reuses the right words with the wrong meaning. The
    bench reports this openly; it is the same trade-off the
    ai-evaluation-platform's citation check makes.
    """
    phrase_tokens = set(content_tokens(phrase))
    if not phrase_tokens:
        return False
    text_tokens = set(stemmed_tokens(text))
    return phrase_tokens <= text_tokens


def extract_citations(text: str) -> list[int]:
    """Citation indices cited in ``text`` as [S1], [S2], ... (1-based)."""
    return [int(m) for m in _CITATION_RE.findall(text)]


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Pure-Python BM25 over ``docs`` for ``query``. Deterministic."""
    tokenized = [tokenize(d) for d in docs]
    n = len(docs)
    if n == 0:
        return []
    avg_len = sum(len(d) for d in tokenized) / n
    df = Counter()
    for tokens in tokenized:
        for tok in set(tokens):
            df[tok] += 1
    query_terms = tokenize(query)
    scores = []
    for tokens in tokenized:
        tf = Counter(tokens)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * (len(tokens) / avg_len if avg_len else 1))
            score += idf * tf[term] * (k1 + 1) / denom
        scores.append(round(score, 4))
    return scores


def question_coverage(sentence: str, question: str) -> float:
    """Fraction of the question's content tokens covered by the sentence.

    Coverage is measured against the *expanded* question (see
    ``expand_question``), so a sentence that answers with a synonym
    still counts as covering the question.
    """
    q_tokens = set(content_tokens(expand_question(question)))
    if not q_tokens:
        return 0.0
    s_tokens = set(stemmed_tokens(sentence))
    return len(q_tokens & s_tokens) / len(q_tokens)
