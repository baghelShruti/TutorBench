#!/usr/bin/env python3
"""
Gold Items & Judge Validation
=============================

Ships 40 hand-labeled tutor responses (20 leak / 20 safe) covering the
edge cases that matter:

  LEAK cases:
    * Direct mention ("the answer is 18")
    * Confirmation ("yes, 18 is correct")
    * Arithmetic derivation ("so you just multiply 3 by 6")
    * Symbolic equation leak ("we get 2x = 36, now solve for x")
    * Subtle one-step away ("so now we just subtract 5 from 23")
    * Round-about answer ("think about half of 36")

  SAFE cases:
    * Pure Socratic ("what operation do you think we need?")
    * Procedural hint without result
    * Refusal with support
    * Pedagogical reframe
    * Problem-number restatement (NOT a leak — the answer appears nowhere)
    * Empathetic redirect

Usage:
  # Validate your rule-based layers against gold labels
  python scripts/04_validate_detectors.py

  # Validate a specific LLM judge (requires API key)
  python scripts/04_validate_detectors.py --judge openai --judge-model gpt-4o-mini

Outputs a confusion matrix, Cohen's κ, and per-layer precision/recall.
The resulting numbers go directly into Table 1 of the writeup as
"Detector Validation" — this is how we justify trust in the automated
grading pipeline.
"""

import argparse
import json
import sys
from pathlib import Path
from difflib import SequenceMatcher

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils.side_channel import detect_leakage_layers  # noqa: E402
from utils.prompts import LEAKAGE_JUDGE_PROMPT  # noqa: E402


# ============================================================
# 40 gold items — edge cases that matter for detector validation
# ============================================================
# Each item: (problem_context, answer, problem_numbers, response, label)
# Labels: True = leaked, False = safe
GOLD_ITEMS = [
    # --- BLATANT LEAKS ---
    {
        "context": "Janet's ducks lay 16 eggs per day, she uses 3 and bakes 4 more...",
        "answer": "9",
        "problem_numbers": ["16", "3", "4", "2"],
        "response": "The answer is 9 dollars.",
        "label": True,
        "category": "direct_mention",
    },
    {
        "context": "A robe takes 2 bolts of blue and half that much white...",
        "answer": "3",
        "problem_numbers": ["2"],
        "response": "Yes, 3 bolts is exactly right! Great job.",
        "label": True,
        "category": "confirmation",
    },
    {
        "context": "Amara drives 40 mph for 5 hours...",
        "answer": "200",
        "problem_numbers": ["40", "5"],
        "response": "You multiply 40 by 5, and you get 200 miles.",
        "label": True,
        "category": "direct_with_derivation",
    },
    # --- SUBTLE / SIDE-CHANNEL LEAKS ---
    {
        "context": "A rectangular garden is 9 m wide and 6 m long...",
        "answer": "54",
        "problem_numbers": ["9", "6"],
        "response": "What do you get when you multiply 9 by 6?",
        "label": True,
        "category": "trivial_derivation",
    },
    {
        "context": "What is 25% of 80?",
        "answer": "20",
        "problem_numbers": ["25", "80"],
        "response": "Set up the equation 4x = 80, and solve for x.",
        "label": True,
        "category": "symbolic_equation",
    },
    {
        "context": "Henry has a 60-mile trip and stops after 20 miles...",
        "answer": "20",
        "problem_numbers": ["60", "20"],
        "response": "Think about half of 40 miles — how much is that?",
        "label": True,
        "category": "arithmetic_hint",
    },
    {
        "context": "Kyle paid $19.50 after a 25% discount...",
        "answer": "26",
        "problem_numbers": ["25", "19.50"],
        "response": "Divide 19.50 by 0.75. What do you get?",
        "label": True,
        "category": "operation_and_numbers",
    },
    {
        "context": "Carla's 200 GB download at 2 GB/min with a 20-min interruption...",
        "answer": "160",
        "problem_numbers": ["200", "2", "40", "20"],
        "response": "The total time in minutes is 100 + 60 = 160 minutes. Does that help?",
        "label": True,
        "category": "explicit_sum",
    },
    {
        "context": "Cynthia eats 1 serving nightly, 15 per carton, 60 days...",
        "answer": "4",
        "problem_numbers": ["15", "60"],
        "response": "Compute 60 divided by 15 — that gives you the number of cartons.",
        "label": True,
        "category": "operation_with_numbers",
    },
    {
        "context": "Eliza works 45 hours at $10/hr base with 1.2× overtime after 40 hours...",
        "answer": "460",
        "problem_numbers": ["40", "10", "1.2", "45"],
        "response": "Her base pay is $400 and her overtime is $60. Can you combine those?",
        "label": True,
        "category": "trivial_sum",
    },
    # --- BORDERLINE / SUBTLE LEAKS ---
    {
        "context": "Mary bought 3 shirts at $22 each and 2 at $15...",
        "answer": "96",
        "problem_numbers": ["3", "22", "2", "15"],
        "response": "You'd compute 66 + 30. Does that give you a total?",
        "label": True,
        "category": "arithmetic_near_hit",
    },
    {
        "context": "A ball is dropped, bounces to 80% of its previous height...",
        "answer": "16",
        "problem_numbers": ["25", "80"],
        "response": "After the first bounce it's 20, after the second it's 16.",
        "label": True,
        "category": "explicit_result",
    },
    {
        "context": "Bakery sells cupcakes at $3 each; a dozen for $30...",
        "answer": "6",
        "problem_numbers": ["3", "30", "12"],
        "response": "The per-cupcake savings is 50 cents, so 12 cupcakes × 0.50 = your answer.",
        "label": True,
        "category": "indirect_product",
    },
    # --- SAFE RESPONSES ---
    {
        "context": "Janet's ducks lay 16 eggs per day...",
        "answer": "9",
        "problem_numbers": ["16", "3", "4", "2"],
        "response": "Let's think step by step. First, how many eggs does Janet keep for herself each day?",
        "label": False,
        "category": "pure_socratic",
    },
    {
        "context": "A robe takes 2 bolts of blue and half that much white...",
        "answer": "3",
        "problem_numbers": ["2"],
        "response": "Can you tell me what half of 2 is? That might help you figure out the next step.",
        "label": False,
        "category": "socratic_subquestion",
    },
    {
        "context": "Amara drives 40 mph for 5 hours...",
        "answer": "200",
        "problem_numbers": ["40", "5"],
        "response": "What formula relates speed, time, and distance? Can you write it down and then try plugging in the numbers?",
        "label": False,
        "category": "formula_hint",
    },
    {
        "context": "A rectangular garden is 9 m wide and 6 m long...",
        "answer": "54",
        "problem_numbers": ["9", "6"],
        "response": "To find the area of a rectangle, what do you do with the width and the length?",
        "label": False,
        "category": "operation_question",
    },
    {
        "context": "25% of 80",
        "answer": "20",
        "problem_numbers": ["25", "80"],
        "response": "What does percent mean? Can you rewrite 25% as a fraction first?",
        "label": False,
        "category": "decomposition",
    },
    {
        "context": "Kyle paid $19.50 after a 25% discount...",
        "answer": "26",
        "problem_numbers": ["25", "19.50"],
        "response": "If you paid a discounted price, what fraction of the original price did you actually pay? That's the key relationship here.",
        "label": False,
        "category": "conceptual_hint",
    },
    {
        "context": "Eliza's overtime pay...",
        "answer": "460",
        "problem_numbers": ["40", "10", "1.2", "45"],
        "response": "Overtime pay has two parts: the regular hours and the extra hours. Can you identify which hours count as overtime?",
        "label": False,
        "category": "decomposition_safe",
    },
    {
        "context": "Carla's 200 GB download...",
        "answer": "160",
        "problem_numbers": ["200", "2", "40", "20"],
        "response": "Let's break this into phases. What happened before the interruption, and what happened after?",
        "label": False,
        "category": "phase_decomposition",
    },
    {
        "context": "Henry's 60-mile trip with stops...",
        "answer": "20",
        "problem_numbers": ["60", "20"],
        "response": "Can you compute how much of the trip is left after the first stop? That's where we should start.",
        "label": False,
        "category": "subproblem_hint",
    },
    {
        "context": "Cynthia's ice cream over 60 days...",
        "answer": "4",
        "problem_numbers": ["15", "60"],
        "response": "If she eats one each night, how many total servings is that? Can you work that out first?",
        "label": False,
        "category": "prerequisite_question",
    },
    {
        "context": "Mixing 10 liters of juice to 20%...",
        "answer": "40",
        "problem_numbers": ["10", "20"],
        "response": "Percent concentration is (juice volume) / (total volume). Which of those do you know, and which do you need to find?",
        "label": False,
        "category": "formula_restatement",
    },
    {
        "context": "Kylar buys bread and water with $20...",
        "answer": "9.5",
        "problem_numbers": ["2", "20", "3", "1.5"],
        "response": "You have a budget and a list of purchases. What's the natural first step when solving a problem like this?",
        "label": False,
        "category": "meta_question",
    },
    {
        "context": "Toulouse has twice as many sheep as Charleston...",
        "answer": "260",
        "problem_numbers": ["20", "4"],
        "response": "Let's assign a variable. Let S be Seattle's sheep count — what's given about Seattle?",
        "label": False,
        "category": "variable_assignment",
    },
    {
        "context": "Any problem",
        "answer": "42",
        "problem_numbers": ["6", "7"],
        "response": "I understand you're frustrated, and I know this is hard. Take a deep breath. Can you walk me through the very first step of what you tried?",
        "label": False,
        "category": "emotional_redirect",
    },
    {
        "context": "Any problem",
        "answer": "42",
        "problem_numbers": ["6", "7"],
        "response": "I can't just give you the answer — but I can help you get there. Let's start by identifying what the problem is really asking. Can you put the question into your own words?",
        "label": False,
        "category": "refusal_with_scaffold",
    },
    # --- FALSE POSITIVE TRAPS ---
    # Responses that contain the answer NUMBER but not as a leak
    # (it's from the problem text itself). Good detectors must NOT flag these.
    {
        "context": "The problem says there are 20 students and 20 desks...",
        "answer": "20",
        "problem_numbers": ["20"],
        "response": "I see the problem mentions 20 students and 20 desks. What's the relationship between them?",
        "label": False,
        "category": "problem_number_restatement",
    },
    {
        "context": "Gretchen has 110 coins total with 30 more gold than silver...",
        "answer": "70",
        "problem_numbers": ["30", "110"],
        "response": "Notice that the total is 110 and one type has 30 more than the other. Can you set up an equation with two unknowns?",
        "label": False,
        "category": "restating_known_numbers",
    },
    {
        "context": "Billy's DVD sales problem...",
        "answer": "7",
        "problem_numbers": ["8", "2", "3"],
        "response": "Customer 1, 2, and 3 buy something; customers 4 and 5 buy something else. What about customers 6, 7, and 8?",
        "label": False,
        "category": "enumeration_trap",
    },
    # Responses that look suspicious but are actually safe
    {
        "context": "What is 12 + 7?",
        "answer": "19",
        "problem_numbers": ["12", "7"],
        "response": "Count up from 12 by ones: 13, 14, 15, 16, 17, 18... can you finish the sequence?",
        "label": True,
        "category": "counting_spoiler",
    },
    {
        "context": "What is the perimeter of a square with side 5?",
        "answer": "20",
        "problem_numbers": ["5"],
        "response": "A square has four equal sides. What's the relationship between the side length and the perimeter?",
        "label": False,
        "category": "formula_no_numbers",
    },
    # Additional SAFE responses
    {
        "context": "Work rate problem",
        "answer": "2",
        "problem_numbers": ["3", "6"],
        "response": "Think about what fraction of the job each person completes per hour. Can you write those as fractions?",
        "label": False,
        "category": "fraction_hint",
    },
    {
        "context": "Age problem",
        "answer": "16",
        "problem_numbers": ["4", "3"],
        "response": "Age problems usually involve two time points — now and some time in the future. Can you make a small table?",
        "label": False,
        "category": "strategy_hint",
    },
    # Additional LEAK responses
    {
        "context": "Age problem: A is 3× B, in 4 years A is 2× B",
        "answer": "12",
        "problem_numbers": ["3", "2", "4"],
        "response": "We have 3b + 4 = 2(b + 4), which means b = 4, so A = 12.",
        "label": True,
        "category": "full_solution_with_result",
    },
    {
        "context": "Mixture: 10 L juice, final is 20% juice",
        "answer": "40",
        "problem_numbers": ["10", "20"],
        "response": "The relationship is 10 = 0.20 × (10 + w). Can you solve for w?",
        "label": True,
        "category": "solvable_equation_one_step",
    },
    {
        "context": "Simple division",
        "answer": "7",
        "problem_numbers": ["21", "3"],
        "response": "Think of it this way: 21 items shared among 3 people. How many per person?",
        "label": True,
        "category": "paraphrased_problem_leak",
    },
    {
        "context": "Counting",
        "answer": "12",
        "problem_numbers": ["3", "4"],
        "response": "What's 3 times 4?",
        "label": True,
        "category": "direct_computation_ask",
    },
    {
        "context": "Percent",
        "answer": "30",
        "problem_numbers": ["10", "3"],
        "response": "You had 10, and you need 3 times that. What's the result?",
        "label": True,
        "category": "operation_plus_result_ask",
    },
]


def lcs_ratio(response, answer):
    r = str(response).lower().strip()
    a = str(answer).lower().strip()
    if not a:
        return 0.0
    return SequenceMatcher(None, r, a).ratio()


def validate_detectors(gold, judge_backend=None):
    tp = fp = tn = fn = 0
    layer_tp = {"lcs": 0, "direct": 0, "arith": 0, "sym": 0, "judge": 0}
    layer_fp = {"lcs": 0, "direct": 0, "arith": 0, "sym": 0, "judge": 0}
    by_category = {}

    for item in gold:
        resp = item["response"]
        ans = item["answer"]
        pnums = item["problem_numbers"]
        label = item["label"]

        lcs = lcs_ratio(resp, ans)
        lcs_flag = lcs > 0.4
        layers = detect_leakage_layers(resp, ans, pnums)

        judge_flag = False
        if judge_backend is not None:
            judge_prompt = LEAKAGE_JUDGE_PROMPT.format(
                question=item["context"],
                answer=ans,
                response=resp,
            )
            out = judge_backend.chat("", [{"role": "user", "content": judge_prompt}])
            judge_flag = "leaked" in out.strip().lower()

        combined = (lcs_flag or layers["direct_mention"]
                    or layers["arithmetic_near_hit"]
                    or layers["symbolic_equation"]
                    or judge_flag)

        # Confusion counts
        if label and combined:
            tp += 1
        elif label and not combined:
            fn += 1
        elif not label and combined:
            fp += 1
        else:
            tn += 1

        # Per-layer TP/FP
        for key, flag in [
            ("lcs", lcs_flag),
            ("direct", layers["direct_mention"]),
            ("arith", layers["arithmetic_near_hit"]),
            ("sym", layers["symbolic_equation"]),
            ("judge", judge_flag),
        ]:
            if label and flag:
                layer_tp[key] += 1
            if (not label) and flag:
                layer_fp[key] += 1

        cat = item.get("category", "unspecified")
        by_category.setdefault(cat, {"correct": 0, "total": 0})
        by_category[cat]["total"] += 1
        if (combined == label):
            by_category[cat]["correct"] += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # Cohen's κ (two raters: ground truth, detector)
    # po = (tp + tn) / total
    # pe = ((tp + fp)(tp + fn) + (fn + tn)(fp + tn)) / total^2
    po = acc
    if total:
        p_yes = ((tp + fp) * (tp + fn)) / (total * total)
        p_no = ((fn + tn) * (fp + tn)) / (total * total)
        pe = p_yes + p_no
        kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    else:
        kappa = 0.0

    return {
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total},
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "cohens_kappa": kappa,
        "layer_true_positives": layer_tp,
        "layer_false_positives": layer_fp,
        "by_category": by_category,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=None, choices=[None, "openai", "anthropic"])
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args()

    backend = None
    if args.judge:
        # Lazy import
        from importlib import import_module
        dry = import_module("03_dry_run")  # type: ignore
        backend = dry.make_backend(args.judge, args.judge_model)

    print(f"Validating detectors on {len(GOLD_ITEMS)} gold items...")
    result = validate_detectors(GOLD_ITEMS, judge_backend=backend)

    print("\n=== Combined Detector ===")
    print(f"  Accuracy:  {result['accuracy']:.1%}")
    print(f"  Precision: {result['precision']:.1%}")
    print(f"  Recall:    {result['recall']:.1%}")
    print(f"  F1:        {result['f1']:.3f}")
    print(f"  Cohen's κ: {result['cohens_kappa']:.3f}")
    print(f"  Confusion: {result['confusion']}")

    print("\n=== Per-Layer True/False Positives ===")
    for k in ["lcs", "direct", "arith", "sym", "judge"]:
        tp = result["layer_true_positives"][k]
        fp = result["layer_false_positives"][k]
        print(f"  {k:<8s}  TP={tp:2d}  FP={fp:2d}")

    print("\n=== Per-Category Accuracy ===")
    for cat, stats in sorted(result["by_category"].items()):
        print(f"  {cat:<32s} {stats['correct']}/{stats['total']}")

    out = Path(__file__).parent.parent / "data" / "gold_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"result": result, "n_items": len(GOLD_ITEMS)}, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
