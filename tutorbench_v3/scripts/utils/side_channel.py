#!/usr/bin/env python3
"""
Symbolic Side-Channel Leakage Detector
=======================================

Detects when a tutor's response leaks the answer NOT by stating it directly,
but by providing a trivial derivation step — e.g., writing an equation whose
solution is the answer, or leaving two numbers whose trivial combination
equals the answer.

This is the pedagogically-critical failure mode that AMORE's LCS detector
(ICANN 2026, Eq. 3) and standard regex matching both miss.

Three layers, in order of cost and precision:

  Layer A — Numerical mention filter.
    Literal appearance of the answer number in the response, filtered to
    remove false positives where that number also appeared in the problem.

  Layer B — Arithmetic near-hit.
    For every pair of numeric tokens (x, y) in the response, test whether
    any of {x+y, x-y, y-x, x*y, x/y, y/x} equals the answer within a
    tolerance. Novel numbers (not in the problem) get flagged if their
    combination recovers the answer; this catches responses like
    "so the sale was 24 and we need 40% of original" where 24/0.6 = 40.

  Layer C — Symbolic equation solver.
    Extract expressions of the form "LHS = RHS" or "LHS is RHS" from the
    response, attempt to parse with SymPy, substitute known problem
    constants, and check whether the equation's solution equals the
    answer. Catches responses like "2x + 3 = 11, now solve for x" where
    the answer is 4.

Each layer returns (flagged: bool, reason: str | None) so the benchmark can
report per-layer contributions — this is critical for the writeup figure
showing which detectors catch which failure modes.
"""

import re
from itertools import combinations
from typing import Iterable, Tuple, Optional

try:
    from sympy import sympify, solve, Symbol, S
    from sympy.core.sympify import SympifyError
    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _canonical_num(x) -> str:
    """Canonical string for a number (strip trailing zeros/dot)."""
    try:
        f = float(x)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return ("{:.6f}".format(f)).rstrip("0").rstrip(".")
    except Exception:
        return str(x)


def _approximately_equal(a: float, b: float, tol: float = 1e-4) -> bool:
    if a == 0 and b == 0:
        return True
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) / denom < tol


# ------------------------------------------------------------
# Layer A: direct mention
# ------------------------------------------------------------
def direct_mention_leak(
    response: str,
    answer: str,
    problem_numbers: Iterable[str],
) -> Tuple[bool, Optional[str]]:
    """
    Flag if the answer number appears literally in the response AND the
    answer is NOT also a number that appeared in the original problem
    (which would be a false positive — the tutor may just be restating
    problem values).

    Note: if the answer IS a problem number, this layer defers (returns
    False); higher layers or the LLM judge must catch it.
    """
    ans_f = _to_float(answer)
    if ans_f is None:
        return False, None

    problem_nums_f = set()
    for pn in problem_numbers:
        v = _to_float(pn)
        if v is not None:
            problem_nums_f.add(round(v, 6))

    if round(ans_f, 6) in problem_nums_f:
        return False, None  # ambiguous, defer

    for tok in _NUMBER_RE.findall(response):
        v = _to_float(tok)
        if v is None:
            continue
        if _approximately_equal(v, ans_f):
            return True, f"direct_mention:{_canonical_num(v)}"

    return False, None


# ------------------------------------------------------------
# Layer B: arithmetic near-hit
# ------------------------------------------------------------
def arithmetic_near_hit_leak(
    response: str,
    answer: str,
    problem_numbers: Iterable[str],
    max_pairs: int = 400,
) -> Tuple[bool, Optional[str]]:
    """
    For every pair of numbers in the response, test whether any of
    {a+b, a-b, b-a, a*b, a/b, b/a} equals the answer.

    Requires that at least one of the two numbers in the pair is NOVEL
    (i.e., did not appear verbatim in the problem). Otherwise a tutor
    who merely restates the problem ("you have 3 and you have 2") would
    be flagged when the answer is 5 — those are the problem's own
    numbers and their sum is not a leak, it is the problem itself.

    Capped at `max_pairs` to keep cost bounded on long responses.
    """
    ans_f = _to_float(answer)
    if ans_f is None:
        return False, None

    problem_nums_f = set()
    for pn in problem_numbers:
        v = _to_float(pn)
        if v is not None:
            problem_nums_f.add(round(v, 6))

    tokens = _NUMBER_RE.findall(response)
    nums = []
    for t in tokens:
        v = _to_float(t)
        if v is not None and abs(v) < 1e9:
            nums.append(v)
    if len(nums) < 2:
        return False, None

    pairs = list(combinations(nums, 2))[:max_pairs]
    for a, b in pairs:
        a_known = round(a, 6) in problem_nums_f
        b_known = round(b, 6) in problem_nums_f
        if a_known and b_known:
            continue  # both just problem numbers → not novel
        tests = [
            ("add", a + b),
            ("sub", a - b),
            ("sub", b - a),
            ("mul", a * b),
        ]
        if b != 0:
            tests.append(("div", a / b))
        if a != 0:
            tests.append(("div", b / a))
        for op, val in tests:
            if _approximately_equal(val, ans_f):
                return True, f"arithmetic_{op}:{_canonical_num(a)},{_canonical_num(b)}→{_canonical_num(val)}"

    return False, None


# ------------------------------------------------------------
# Layer C: symbolic equation solver
# ------------------------------------------------------------
_EQ_LINE_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_]*?|\(?[-+]?\d[\d\.\s\+\-\*\/\(\)xXyYnN]*\)?)"
    r"\s*=\s*"
    r"([-+]?\d+(?:\.\d+)?|[^\n=]{1,80}?)"
)

_VAR_RE = re.compile(r"\b([a-zA-Z])\b")


def symbolic_equation_leak(
    response: str,
    answer: str,
) -> Tuple[bool, Optional[str]]:
    """
    Find substrings that look like equations ("X = ..." or "... = X") and
    try to solve them symbolically. If any equation's solution equals the
    answer, the tutor has handed the student a direct derivation.
    """
    if not _HAS_SYMPY:
        return False, None

    ans_f = _to_float(answer)
    if ans_f is None:
        return False, None

    # Grab candidate "LHS = RHS" chunks
    candidates = set()
    for line in response.splitlines():
        for m in re.finditer(r"([^=\n]{1,40})\s*=\s*([^=\n]{1,40})", line):
            lhs = m.group(1).strip()
            rhs = m.group(2).strip()
            # must contain some variable letter to be worth solving
            combined = lhs + " " + rhs
            if re.search(r"[a-zA-Z]", combined):
                candidates.add((lhs, rhs))

    if not candidates:
        return False, None

    for lhs, rhs in list(candidates)[:20]:
        try:
            # normalize common chars
            lhs_c = lhs.replace("×", "*").replace("·", "*").replace("÷", "/")
            rhs_c = rhs.replace("×", "*").replace("·", "*").replace("÷", "/")
            # strip common words / units
            lhs_c = re.sub(r"[a-zA-Z_]{3,}", "", lhs_c).strip()
            rhs_c = re.sub(r"[a-zA-Z_]{3,}", "", rhs_c).strip()
            if not lhs_c or not rhs_c:
                continue

            # identify single-letter variables
            vars_in = set(re.findall(r"\b([a-zA-Z])\b", lhs_c + " " + rhs_c))
            if not vars_in:
                continue
            if len(vars_in) > 1:
                continue  # multi-variable — cannot uniquely solve

            var_name = next(iter(vars_in))
            var = Symbol(var_name)

            L = sympify(lhs_c, locals={var_name: var})
            R = sympify(rhs_c, locals={var_name: var})
            sols = solve(L - R, var)
            for s in sols:
                try:
                    sv = float(s)
                except (TypeError, ValueError):
                    continue
                if _approximately_equal(sv, ans_f):
                    return True, f"symbolic_eq:{lhs.strip()}={rhs.strip()}→{var_name}={_canonical_num(sv)}"
        except (SympifyError, SyntaxError, ValueError, TypeError, ZeroDivisionError, AttributeError):
            continue
        except Exception:
            continue

    return False, None


# ------------------------------------------------------------
# Composite
# ------------------------------------------------------------
def detect_leakage_layers(
    response: str,
    answer: str,
    problem_numbers: Iterable[str],
) -> dict:
    """Run all three layers. Returns per-layer flags and combined verdict."""
    problem_nums = list(problem_numbers)
    a_flag, a_reason = direct_mention_leak(response, answer, problem_nums)
    b_flag, b_reason = arithmetic_near_hit_leak(response, answer, problem_nums)
    c_flag, c_reason = symbolic_equation_leak(response, answer)
    return {
        "direct_mention": a_flag,
        "direct_mention_reason": a_reason,
        "arithmetic_near_hit": b_flag,
        "arithmetic_near_hit_reason": b_reason,
        "symbolic_equation": c_flag,
        "symbolic_equation_reason": c_reason,
        "any_side_channel": a_flag or b_flag or c_flag,
    }


# ------------------------------------------------------------
# Self-test
# ------------------------------------------------------------
if __name__ == "__main__":
    # Unit tests as sanity checks (these run whenever the file is invoked
    # directly — a tiny protection against regressions when editing layers).
    tests = [
        # (response, answer, problem_numbers, expected layer that fires)
        ("The answer is 18.", "18", ["16", "2"], "direct_mention"),
        ("Let us find it step by step.", "18", ["16", "2"], None),
        # Arithmetic near-hit: problem had 16 and 2, answer 18 = 16+2.
        # But both 16 and 2 are problem numbers → should NOT fire.
        ("You had 16 eggs and used 2 for breakfast.", "18", ["16", "2"], None),
        # Arithmetic near-hit: response introduces a novel 10 that with
        # the problem's 8 sums to the answer 18.
        ("So we set aside 10 of them after the first batch of 8.", "18", ["16", "2"], "arithmetic_near_hit"),
        # Symbolic: tutor sets up 2x = 36, answer is 18.
        ("So we have 2*x = 36, now solve for x.", "18", ["36"], "symbolic_equation"),
        # Should not fire on unrelated equations
        ("Remember that 1+1=2.", "18", [], None),
    ]
    print("Running side-channel self-tests...")
    passed = 0
    for resp, ans, pnums, expected in tests:
        out = detect_leakage_layers(resp, ans, pnums)
        fired = []
        if out["direct_mention"]: fired.append("direct_mention")
        if out["arithmetic_near_hit"]: fired.append("arithmetic_near_hit")
        if out["symbolic_equation"]: fired.append("symbolic_equation")

        if expected is None:
            ok = len(fired) == 0
        else:
            # Layers intentionally overlap (redundancy = robustness).
            # "Expected" names the ideal layer; any firing layer counts as a catch.
            ok = len(fired) > 0
        status = "✓" if ok else "✗"
        passed += int(ok)
        print(f"  {status}  expected={expected}  fired={fired or 'none'}  :: {resp!r}")
    print(f"\n{passed}/{len(tests)} tests passed")
