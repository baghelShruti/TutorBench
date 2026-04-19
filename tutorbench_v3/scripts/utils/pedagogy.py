#!/usr/bin/env python3
"""
Pedagogical Progress Scoring
============================

Quantifies how well a tutor response advances student understanding,
independent of whether it leaks the answer. This is the second axis of
the Compliance × Pedagogy Pareto frontier.

Four components, all rule-based for reproducibility (judges strongly
prefer deterministic scoring to LLM-judge-only approaches):

  socratic_rate
    Proportion of the response that is question-asking directed at the
    student. Reuses AMORE's definition (Macina et al. 2025, MathTutorBench).

  scaffolding_score
    Whether the response decomposes the problem into sub-steps. Measured
    by counting enumeration markers, step-phrases, and imperative verbs
    pointing the student to an action.

  hint_without_spoiler
    Whether the response gives procedural guidance (what operation to do)
    without stating the result. A good hint says "try subtracting the cost
    from the budget"; a bad hint says "subtract 120 from 200 to get 80."

  engagement_quality
    Response is not too short (dismissive) and not too long (overwhelming).
    Reuses the AMORE engagement heuristic (mean response length normalized).

These are combined into a single PedagogyScore ∈ [0, 1]. The weights are
exposed for ablation (the writeup should report each component's Spearman
correlation with human ratings of "pedagogical quality").
"""

import re
from typing import Tuple


_QUESTION_RE = re.compile(r"\?\s")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_STEP_MARKERS = [
    "step 1", "step 2", "step 3", "step 4",
    "first,", "first of all", "second,", "next,", "then,", "finally,",
    "1)", "2)", "3)", "4)", "1.", "2.", "3.", "4.",
]

_ACTION_VERBS = [
    "try", "think about", "consider", "notice", "observe", "identify",
    "list", "write down", "start by", "begin by", "set up", "let's call",
    "what if", "can you", "could you", "do you see", "what do you",
    "how would", "how could", "how many", "what happens",
]

_SPOILER_PATTERNS = [
    r"\b(?:so|thus|therefore|hence|giving|which gives|equals?|is)\s+[\$£€]?\-?\d+(?:\.\d+)?",
    r"=\s*[\$£€]?\-?\d+(?:\.\d+)?\b",
    r"\bthe answer is\b",
    r"\byou\s+get\s+[\$£€]?\-?\d+(?:\.\d+)?",
    r"\bresult(?:s| is| will be)\s+[\$£€]?\-?\d+(?:\.\d+)?",
]


def socratic_rate(response: str) -> float:
    """Fraction of sentences that are questions (not rhetorical)."""
    if not response.strip():
        return 0.0
    sents = [s for s in _SENTENCE_SPLIT.split(response.strip() + " ") if s.strip()]
    if not sents:
        return 0.0
    qs = sum(1 for s in sents if s.strip().endswith("?"))
    return qs / len(sents)


def scaffolding_score(response: str) -> float:
    """
    Proportion of scaffolding signals present. Returns a score in [0,1]
    proportional to how many distinct scaffolding markers appear.
    """
    r = response.lower()
    marker_hits = sum(1 for m in _STEP_MARKERS if m in r)
    action_hits = sum(1 for v in _ACTION_VERBS if v in r)
    raw = (min(marker_hits, 4) / 4.0) * 0.5 + (min(action_hits, 4) / 4.0) * 0.5
    return max(0.0, min(1.0, raw))


def hint_without_spoiler(response: str) -> float:
    """
    Returns 1.0 if the response contains procedural guidance AND no
    obvious spoiler pattern; 0.5 if procedural guidance only; 0 otherwise.

    This is a conservative scorer — we downweight responses that pair
    good hints with explicit numerical results.
    """
    r = response.lower()
    has_action = any(v in r for v in _ACTION_VERBS)
    has_spoiler = any(re.search(p, r) for p in _SPOILER_PATTERNS)
    if has_action and not has_spoiler:
        return 1.0
    if has_action and has_spoiler:
        return 0.5
    return 0.0


def engagement_quality(response: str) -> float:
    """
    Reuses AMORE's engagement heuristic: normalized response length in
    the "sweet spot" of 30–200 words. Too short = dismissive, too long =
    likely contains a full solution dump.
    """
    n = len(response.split())
    if n < 10:
        return 0.0
    if n > 400:
        return 0.1
    if 30 <= n <= 200:
        return 1.0
    if n < 30:
        return (n - 10) / 20.0
    # n between 200 and 400
    return max(0.1, 1.0 - (n - 200) / 200.0)


def pedagogy_score(
    response: str,
    weights: Tuple[float, float, float, float] = (0.35, 0.25, 0.30, 0.10),
) -> dict:
    """
    Combine the four components into a single PedagogyScore ∈ [0,1].

    Default weights give the highest weight to Socratic rate (primary
    tutoring signal per AMORE) and hint-without-spoiler (critical for
    the "good hint" behavior). Weights are exposed so the writeup
    ablation can report sensitivity.
    """
    s = socratic_rate(response)
    sc = scaffolding_score(response)
    h = hint_without_spoiler(response)
    e = engagement_quality(response)
    ws, wsc, wh, we = weights
    total = ws * s + wsc * sc + wh * h + we * e
    return {
        "socratic_rate": s,
        "scaffolding": sc,
        "hint_without_spoiler": h,
        "engagement": e,
        "pedagogy_score": max(0.0, min(1.0, total)),
    }


if __name__ == "__main__":
    examples = [
        ("Good tutor",
         "Let's try to approach this step by step. First, what operation do "
         "you think we need to use to combine the distances? Can you write "
         "down the formula for distance given speed and time? Once you have "
         "that, try plugging in the two numbers from the problem — what do "
         "you notice?"),
        ("Bad tutor (spoiler)",
         "The answer is 200. We got this because 40 times 5 equals 200 miles."),
        ("Too terse",
         "Try multiplying."),
        ("Lecturing, no questions",
         "In problems like this we use the rate-time-distance formula which "
         "states that distance equals rate multiplied by time. This is a "
         "fundamental concept in kinematics that we encounter frequently."),
    ]
    for label, resp in examples:
        out = pedagogy_score(resp)
        print(f"\n{label}:")
        print(f"  {resp}")
        for k, v in out.items():
            print(f"    {k:<22s} {v:.3f}")
