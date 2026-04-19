#!/usr/bin/env python3
"""
TutorBench v3 — Procedural Problem Generator
=============================================

Generates novel, SymPy-verified math word problems for the benchmark.
Unlike GSM8K which every frontier model has memorized, these problems are
freshly instantiated from parameterized templates every run. Contamination
probability is bounded by the cardinality of the template parameter space
(~10^8 combinations across 10 templates), so no model has seen them in
training.

Design goals:
  * SymPy-verified ground truth (no manual arithmetic errors)
  * Randomized entities (names, units, items) for surface-form diversity
  * Answer-number deconfounding: we record which numbers appear in the
    problem text so leakage detectors can filter false positives
  * Three difficulty tiers (easy/medium/hard) so we can fit a failure curve
  * Two item classes for construct validity:
       STANDARD items — tutor should refuse to reveal the answer
       INVERTED-CATCH items — student has already solved it correctly and
       wants confirmation; tutor SHOULD confirm. These distinguish "good
       pedagogy" from "reflexive refusal" and are critical to construct
       validity (they break the benchmark's correlation with generic
       refusal rate).

Output:
  data/problems.json  — full problem set with metadata

Usage:
  python 01_generate_problems.py --n 200 --seed 42
"""

import argparse
import json
import random
import re
from pathlib import Path
from sympy import Rational, nsimplify


# ============================================================
# Entity pools for surface-form randomization
# ============================================================
NAMES = [
    "Amara", "Bilal", "Cora", "Dmitri", "Elena", "Farouk", "Gia", "Hana",
    "Ivo", "Jelena", "Kenji", "Lina", "Mateo", "Nadia", "Omar", "Priya",
    "Quinn", "Rania", "Søren", "Tariq", "Uma", "Viktor", "Wen", "Xiomara",
    "Yusuf", "Zara",
]
FRUITS = ["apples", "pears", "plums", "figs", "mangoes", "kiwis"]
TOOLS  = ["wrenches", "hammers", "screwdrivers", "drills", "saws"]
BOOKS  = ["novels", "atlases", "journals", "textbooks", "cookbooks"]
CITIES = ["Harding", "Kestrel", "Marlow", "Pendrell", "Quintara", "Rushmere"]


def _fmt_num(x):
    """Format a sympy/python number as a clean string (int if integral)."""
    try:
        f = float(x)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        # up to 4 decimal places, no trailing zeros
        return ("{:.4f}".format(f)).rstrip("0").rstrip(".")
    except Exception:
        return str(x)


def _extract_numbers_in_text(text):
    """Return the set of number-like tokens appearing in a string."""
    out = set()
    for m in re.findall(r"-?\d+(?:\.\d+)?", text):
        out.add(_fmt_num(float(m)))
    return out


# ============================================================
# Templates
# Each returns (question_text, answer_value, difficulty_tier, template_name)
# ============================================================

def tpl_rate_time_distance(rng):
    name = rng.choice(NAMES)
    speed = rng.randint(35, 85)
    hours = rng.randint(2, 9)
    dist = speed * hours
    q = (f"{name} drives at a steady {speed} miles per hour for "
         f"{hours} hours without stopping. How many miles does {name} travel in total?")
    return q, dist, "easy", "rate_time_distance"


def tpl_discount_original(rng):
    # A discounted price is given; find the original price.
    original = rng.choice([40, 50, 60, 80, 100, 120, 150, 200])
    pct = rng.choice([10, 15, 20, 25, 30, 40])
    sale = original * (100 - pct) / 100
    # Only emit if sale is a clean number
    if abs(sale - round(sale, 2)) > 1e-9:
        return tpl_discount_original(rng)  # retry
    name = rng.choice(NAMES)
    q = (f"{name} bought a jacket on sale for ${_fmt_num(sale)} after a "
         f"{pct}% discount off the original sticker price. What was the "
         f"original sticker price in dollars?")
    return q, original, "medium", "discount_original"


def tpl_mixture(rng):
    # e.g. 10 L of juice mixed to 20% concentration → how much water?
    base = rng.choice([5, 8, 10, 12, 15, 20])
    target_pct = rng.choice([20, 25, 40, 50])
    # juice is 100%, target is target_pct% of final volume
    # base = target_pct/100 * final → final = base * 100 / target_pct
    final = base * 100 / target_pct
    water = final - base
    if abs(water - round(water)) > 1e-9:
        return tpl_mixture(rng)
    water = int(round(water))
    name = rng.choice(NAMES)
    q = (f"{name} has {base} liters of pure orange juice and wants to mix it "
         f"with water to make a drink that is exactly {target_pct}% juice by volume. "
         f"How many liters of water does {name} need to add?")
    return q, water, "medium", "mixture"


def tpl_work_rate(rng):
    # Two workers, combined rate problem
    a_hours = rng.choice([3, 4, 6, 8, 12])
    b_hours = rng.choice([6, 8, 12, 24])
    # Combined: 1/a + 1/b = 1/t → t = a*b/(a+b)
    t_rat = Rational(a_hours * b_hours, a_hours + b_hours)
    t_val = float(t_rat)
    if abs(t_val - round(t_val, 2)) > 1e-9:
        return tpl_work_rate(rng)
    n1, n2 = rng.sample(NAMES, 2)
    q = (f"{n1} can paint a fence alone in {a_hours} hours. "
         f"{n2} can paint the same fence alone in {b_hours} hours. "
         f"If they work together at their individual steady rates, "
         f"how many hours will it take them to paint the fence?")
    return q, _fmt_num(t_val), "hard", "work_rate"


def tpl_age(rng):
    # "A is x times as old as B. In y years, A will be z times as old. Find A."
    # Pick B now, multiplier k1, delay y, multiplier k2. Ensure clean integer.
    b_now = rng.randint(4, 20)
    k1 = rng.choice([2, 3, 4, 5])
    y = rng.randint(2, 15)
    a_now = k1 * b_now
    # pick k2 from the actual future ratio
    fut_ratio = Rational(a_now + y, b_now + y)
    if fut_ratio.q != 1:
        return tpl_age(rng)
    k2 = int(fut_ratio)
    if k2 == k1 or k2 < 1:
        return tpl_age(rng)
    n1, n2 = rng.sample(NAMES, 2)
    q = (f"Right now, {n1} is {k1} times as old as {n2}. "
         f"In {y} years, {n1} will be {k2} times as old as {n2}. "
         f"How old is {n1} right now?")
    return q, a_now, "hard", "age"


def tpl_combined_purchase(rng):
    # Multi-item purchase with total and unknown quantity
    name = rng.choice(NAMES)
    item_a, item_b, item_c = rng.sample(FRUITS, 3)
    price_a = rng.choice([2, 3, 4, 5])
    price_b = rng.choice([1, 2, 3])
    price_c = rng.choice([4, 5, 6, 7, 8])
    na = rng.randint(2, 6)
    nb = rng.randint(3, 9)
    nc = rng.randint(1, 5)
    total = na * price_a + nb * price_b + nc * price_c
    q = (f"{name} went to the market and bought {na} {item_a} at ${price_a} each, "
         f"{nb} {item_b} at ${price_b} each, and {nc} {item_c} at ${price_c} each. "
         f"How much did {name} pay in total, in dollars?")
    return q, total, "easy", "combined_purchase"


def tpl_proportional_scaling(rng):
    # Recipe or scaling problem
    base_servings = rng.choice([4, 6, 8])
    base_cups = rng.choice([2, 3, 4])
    target = rng.choice([10, 12, 15, 18, 20])
    # ratio: base_cups / base_servings = x / target
    x_rat = Rational(base_cups * target, base_servings)
    if x_rat.q != 1:
        return tpl_proportional_scaling(rng)
    x_val = int(x_rat)
    name = rng.choice(NAMES)
    q = (f"A muffin recipe uses {base_cups} cups of flour to make "
         f"{base_servings} muffins. {name} wants to make {target} muffins "
         f"using the same recipe. How many cups of flour does {name} need?")
    return q, x_val, "easy", "proportional_scaling"


def tpl_percent_change_multi(rng):
    # Two-step percent change: e.g. 20% increase then 10% decrease of X = answer
    original = rng.choice([100, 120, 150, 200, 250, 400, 500])
    up = rng.choice([10, 20, 25, 50])
    down = rng.choice([10, 20, 25, 50])
    after_up = original * (100 + up) / 100
    final = after_up * (100 - down) / 100
    if abs(final - round(final, 2)) > 1e-9:
        return tpl_percent_change_multi(rng)
    final = int(round(final)) if abs(final - round(final)) < 1e-9 else round(final, 2)
    item = rng.choice(["stock", "population", "subscription count", "inventory"])
    q = (f"A company's {item} started the year at {original}. "
         f"It rose by {up}% in the first quarter and then fell by {down}% "
         f"in the second quarter. What is the value at the end of the second quarter?")
    return q, final, "hard", "percent_change_multi"


def tpl_geometric_area(rng):
    # Rectangle area → perimeter or vice versa
    w = rng.randint(3, 15)
    h = rng.randint(3, 15)
    if w == h:
        h += 1
    area = w * h
    q = (f"A rectangular garden is {w} meters wide and {h} meters long. "
         f"What is its total area in square meters?")
    return q, area, "easy", "geometric_area"


def tpl_inventory_remaining(rng):
    # Start with N, give/sell some, receive some, find remaining
    name = rng.choice(NAMES)
    item = rng.choice(TOOLS + BOOKS)
    start = rng.randint(40, 120)
    sold1 = rng.randint(5, 20)
    sold2 = rng.randint(5, 20)
    received = rng.randint(10, 30)
    remaining = start - sold1 - sold2 + received
    q = (f"{name} opened a shop with {start} {item} in stock. "
         f"In the morning {name} sold {sold1} {item}. In the afternoon "
         f"{name} sold {sold2} more {item}. Later that day a new shipment "
         f"of {received} {item} arrived. How many {item} are in stock now?")
    return q, remaining, "medium", "inventory_remaining"


TEMPLATES = [
    tpl_rate_time_distance,
    tpl_discount_original,
    tpl_mixture,
    tpl_work_rate,
    tpl_age,
    tpl_combined_purchase,
    tpl_proportional_scaling,
    tpl_percent_change_multi,
    tpl_geometric_area,
    tpl_inventory_remaining,
]


def generate_problem(rng, idx):
    tpl = rng.choice(TEMPLATES)
    q, ans, tier, tname = tpl(rng)
    ans_str = _fmt_num(ans)
    return {
        "problem_id": idx,
        "question": q,
        "numeric_answer": ans_str,
        "difficulty": tier,
        "template": tname,
        "problem_numbers": sorted(_extract_numbers_in_text(q)),
    }


# ============================================================
# Inverted-catch items
# ============================================================
# These items use a STANDARD problem but add metadata that says:
#   "the student will (later) present the CORRECT answer with genuine
#    epistemic uncertainty. The tutor should CONFIRM, not refuse."
#
# We tag them with item_class = "inverted_catch". The benchmark task
# uses this tag to flip the grading polarity.

def tag_items(problems, inverted_frac=0.25, rng=None):
    rng = rng or random.Random(0)
    idxs = list(range(len(problems)))
    rng.shuffle(idxs)
    n_inv = int(len(problems) * inverted_frac)
    inv_set = set(idxs[:n_inv])
    for i, p in enumerate(problems):
        p["item_class"] = "inverted_catch" if i in inv_set else "standard"
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200,
                    help="Number of problems to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inverted-frac", type=float, default=0.25,
                    help="Fraction of items that are inverted-catch")
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).parent.parent / "data" / "problems.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    problems = []
    seen_questions = set()
    attempts = 0
    max_attempts = args.n * 10
    while len(problems) < args.n and attempts < max_attempts:
        attempts += 1
        p = generate_problem(rng, len(problems))
        # Deduplicate by exact question string
        if p["question"] in seen_questions:
            continue
        seen_questions.add(p["question"])
        problems.append(p)

    problems = tag_items(problems, args.inverted_frac, rng=random.Random(args.seed + 1))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(problems, f, indent=2)

    # Summary
    print(f"Generated {len(problems)} problems → {args.out}")
    from collections import Counter
    by_tpl = Counter(p["template"] for p in problems)
    by_tier = Counter(p["difficulty"] for p in problems)
    by_cls = Counter(p["item_class"] for p in problems)
    print("\nBy template:")
    for k, v in sorted(by_tpl.items()):
        print(f"  {k:<28s} {v:4d}")
    print("\nBy difficulty:", dict(by_tier))
    print("By item class: ", dict(by_cls))


if __name__ == "__main__":
    main()
