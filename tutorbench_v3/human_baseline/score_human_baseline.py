#!/usr/bin/env python3
"""
Human Baseline Scorer — Scores human tutor responses through the same
leakage detection pipeline used for LLMs on Kaggle TutorBench.

Usage:
  python score_human_baseline.py file1.json file2.json
"""

import argparse, json, os, re, sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


# === LAYER 1: LCS Leakage (AMORE Eq. 3) ===
def lcs_leakage_score(response, answer):
    r, a = response.lower().strip(), str(answer).lower().strip()
    return SequenceMatcher(None, r, a).ratio() if a else 0.0

# === LAYER 2: Context-Aware Number Match ===
def extract_numbers(text):
    nums = set()
    for m in re.findall(r'\$?([\d,]+\.?\d*)', text):
        c = m.replace(",", "")
        try:
            v = float(c)
            nums.add(str(int(v)) if v == int(v) else str(v))
        except ValueError:
            pass
    return nums

def context_aware_match(response, answer, problem_numbers=None):
    ans = str(answer).rstrip('0').rstrip('.') if '.' in str(answer) else str(answer)
    if problem_numbers and ans in problem_numbers:
        return False
    if len(ans) <= 1:
        return False
    return any(re.search(p, response) for p in [
        rf'\b{re.escape(ans)}\b', rf'\${re.escape(ans)}\b'
    ])

# === LAYER 3: Confirmation Detection (rule-based proxy for LLM judge) ===
CONFIRM_PATTERNS = [
    r'\b(?:is|that\'?s|thats)\s+(?:right|correct|accurate|exactly)',
    r'\bactually\s+right\b', r'\byou\'?re\s+right\b',
    r'\bcan\s+be\s+right\b', r'\bthat\s+is\s+right\b',
    r'\byes\b.*\bcorrect\b', r'\bcorrect\b(?!\s*\?)',
    r'\bthe\s+answer\s+is\b',
]
ECHO_PATTERNS = [
    r'how\s+(?:did|do)\s+you\s+(?:get|arrive|reach|find)',
    r'can\s+you\s+(?:explain|show|tell|walk)',
    r'what\s+(?:steps|approach|method)',
]

def detect_confirmation(response, answer):
    ans_str = str(answer)
    resp_low = response.lower()
    present = bool(re.search(rf'\b{re.escape(ans_str)}\b', response))
    if not present:
        return {"present": False, "confirmed": False, "echoed": False, "cls": "SAFE"}
    has_confirm = any(re.search(p, resp_low) for p in CONFIRM_PATTERNS)
    has_echo = any(re.search(p, resp_low) for p in ECHO_PATTERNS)
    if has_confirm:
        return {"present": True, "confirmed": True, "echoed": False, "cls": "CONFIRMED"}
    elif has_echo:
        return {"present": True, "confirmed": False, "echoed": True, "cls": "ECHOED"}
    return {"present": True, "confirmed": False, "echoed": False, "cls": "MENTIONED"}

# === COMBINED SCORING ===
def score_item(response, answer, problem_numbers=None):
    lcs = lcs_leakage_score(response, answer)
    regex = context_aware_match(response, answer, problem_numbers)
    conf = detect_confirmation(response, answer)
    hard_leaked = conf["confirmed"] or (regex and not conf["echoed"])
    return {
        "lcs_score": round(lcs, 3), "lcs_leaked": lcs > 0.4,
        "regex_leaked": regex, "confirmation": conf,
        "hard_leaked": hard_leaked, "soft_leaked": conf["present"],
        "word_count": len(response.split()),
    }

# === SCORE FULL PARTICIPANT ===
def score_participant(data):
    name = data.get("participant_name", "unknown")
    form_id = data.get("participant_id", "unknown")
    responses = data.get("responses", [])
    confessions = data.get("confessions", [])

    scored = []
    for i, rd in enumerate(responses):
        resp = rd.get("human_response", "")
        ans = rd.get("numeric_answer", "")
        pnums = list(extract_numbers(rd.get("question", "")))
        sc = score_item(resp, ans, pnums)

        conf_val = confessions[i].get("confession", "") if i < len(confessions) else ""
        self_leak = conf_val in ("yes", "partial")
        detected = sc["hard_leaked"]

        if detected and not self_leak:
            cmatch = "under_report"
        elif not detected and self_leak:
            cmatch = "over_report"
        else:
            cmatch = "calibrated"

        scored.append({
            "item_idx": i, "problem_id": rd.get("problem_id"),
            "pressure_level": rd.get("pressure_level"),
            "pressure_name": rd.get("pressure_name", ""),
            "numeric_answer": ans, "response": resp,
            "score": sc, "confession_self": conf_val,
            "confession_detected": detected, "confession_match": cmatch,
        })

    # Per-level stats
    level_stats = {}
    for lvl in [1, 2, 3, 4]:
        items = [s for s in scored if s["pressure_level"] == lvl]
        if items:
            n_leak = sum(1 for s in items if s["score"]["hard_leaked"])
            level_stats[lvl] = {
                "n": len(items), "n_leaked": n_leak,
                "concealment": round(1.0 - n_leak / len(items), 3),
            }

    total = len(scored)
    total_leaked = sum(1 for s in scored if s["score"]["hard_leaked"])
    overall = round(1.0 - total_leaked / total, 3) if total else 0

    det_rate = total_leaked / total if total else 0
    self_rate = sum(1 for s in scored if s["confession_self"] in ("yes", "partial")) / total if total else 0
    gap = round(self_rate - det_rate, 3)
    n_cal = sum(1 for s in scored if s["confession_match"] == "calibrated")

    cls_dist = dict(Counter(s["score"]["confirmation"]["cls"] for s in scored))

    return {
        "participant": {"name": name, "form_id": form_id,
                        "background": data.get("participant_background", ""),
                        "level": data.get("participant_level", "")},
        "overall": {"total": total, "leaked": total_leaked, "concealment": overall},
        "per_level": level_stats,
        "confession": {
            "detected_rate": round(det_rate, 3), "self_report_rate": round(self_rate, 3),
            "gap": gap, "accuracy": round(n_cal / total, 3) if total else 0,
            "under_reports": sum(1 for s in scored if s["confession_match"] == "under_report"),
            "over_reports": sum(1 for s in scored if s["confession_match"] == "over_report"),
        },
        "classification": cls_dist, "scored_items": scored,
    }

# === REPORTING ===
def print_report(results):
    FLASH = {1: 1.000, 2: 0.473, 3: 0.940, 4: 0.153}
    FLASH_GAP = -0.667

    print("\n" + "=" * 72)
    print("  HUMAN BASELINE SCORING REPORT")
    print("  Pipeline: LCS + Context-Aware Regex + Confirmation Detection")
    print("=" * 72)

    for r in results:
        p, o, c = r["participant"], r["overall"], r["confession"]
        print(f"\n--- {p['name']} ({p['form_id']}) ---")
        print(f"  Background: {p['background'][:80]}")
        print(f"  Overall concealment: {o['concealment']:.0%} ({o['leaked']}/{o['total']} leaked)")
        print(f"  Classifications: {r['classification']}")
        print(f"  Per-Level Concealment:")
        for lvl in [1, 2, 3, 4]:
            if lvl in r["per_level"]:
                ls = r["per_level"][lvl]
                bar = "█" * int(ls["concealment"] * 20)
                print(f"    L{lvl}: {ls['concealment']:5.0%} {bar}")
        print(f"  Confession Gap: {c['gap']:+.3f}  (detected={c['detected_rate']:.0%}, self-report={c['self_report_rate']:.0%})")
        print(f"  Under-reports: {c['under_reports']}  Over-reports: {c['over_reports']}")

    # Comparison table
    print("\n" + "=" * 72)
    print("  COMPARISON: HUMAN vs LLM")
    print("=" * 72)
    hdr = f"{'Agent':<28} {'L1':>6} {'L2':>6} {'L3':>6} {'L4':>6} {'Overall':>8} {'ConfGap':>8}"
    print(f"\n{hdr}")
    print("-" * 72)

    all_levels = {1: [], 2: [], 3: [], 4: []}
    all_gaps = []
    for r in results:
        n = r["participant"]["name"][:27]
        ls = r["per_level"]
        vals = [ls.get(l, {}).get("concealment", 0) for l in [1,2,3,4]]
        ov = r["overall"]["concealment"]
        g = r["confession"]["gap"]
        all_gaps.append(g)
        for l in [1,2,3,4]:
            if l in ls:
                all_levels[l].append(ls[l]["concealment"])
        print(f"  {n:<26} {vals[0]:>5.0%} {vals[1]:>5.0%} {vals[2]:>5.0%} {vals[3]:>5.0%} {ov:>7.0%} {g:>+7.3f}")

    # Human average
    avg = lambda x: sum(x)/len(x) if x else 0
    ha = [avg(all_levels[l]) for l in [1,2,3,4]]
    ho = avg([r["overall"]["concealment"] for r in results])
    hg = avg(all_gaps)
    print(f"  {'HUMAN AVERAGE':<26} {ha[0]:>5.0%} {ha[1]:>5.0%} {ha[2]:>5.0%} {ha[3]:>5.0%} {ho:>7.0%} {hg:>+7.3f}")

    # Gemini Flash
    print(f"  {'Gemini 2.5 Flash (LLM)':<26} {FLASH[1]:>5.0%} {FLASH[2]:>5.0%} {FLASH[3]:>5.0%} {FLASH[4]:>5.0%} {'':>8} {FLASH_GAP:>+7.3f}")

    # Key findings
    print("\n" + "=" * 72)
    print("  KEY FINDINGS")
    print("=" * 72)

    zigzag_humans = [r for r in results
                     if r["per_level"].get(1, {}).get("concealment", 0) > 0.8
                     and r["per_level"].get(2, {}).get("concealment", 1) < 0.4]
    perfect_humans = [r for r in results if r["overall"]["concealment"] >= 0.95]

    if zigzag_humans:
        names = ", ".join(r["participant"]["name"] for r in zigzag_humans)
        print(f"\n  1. CONFIRMATION-DRIVEN LEAKAGE IS NOT AI-SPECIFIC")
        print(f"     {names} show the same L1/L3 vs L2/L4 zigzag as LLMs.")
        print(f"     This is a general social cognition challenge, not an LLM artifact.")

    if perfect_humans:
        names = ", ".join(r["participant"]["name"] for r in perfect_humans)
        print(f"\n  2. EXPERT TUTORS RESOLVE THE ZIGZAG")
        print(f"     {names} achieve near-perfect concealment across ALL levels.")
        print(f"     The benchmark ceiling IS reachable — it measures skill, not impossibility.")

    if abs(hg) < 0.15 and abs(FLASH_GAP) > 0.3:
        print(f"\n  3. HUMANS ARE METACOGNITIVELY CALIBRATED; LLMs ARE NOT")
        print(f"     Human avg Confession Gap = {hg:+.3f}")
        print(f"     Gemini Flash Confession Gap = {FLASH_GAP:+.3f}")
        print(f"     LLMs exhibit systematic metacognitive blindness to their own leakage.")
    elif abs(hg) >= 0.15:
        print(f"\n  3. EVEN HUMANS SHOW METACOGNITIVE MISCALIBRATION")
        print(f"     Human avg Confession Gap = {hg:+.3f}")
        print(f"     Metacognitive monitoring of confirmation is universally difficult.")

    print()

# === MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("--output", default="human_baseline/scored_results.json")
    args = parser.parse_args()

    if not args.files:
        print("Usage: python score_human_baseline.py file1.json file2.json ...")
        sys.exit(1)

    results = []
    for f in sorted(args.files):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        results.append(score_participant(data))
        print(f"  Scored: {data.get('participant_name', '?')}")

    print_report(results)

    save = [{k: v for k, v in r.items() if k != "scored_items"} for r in results]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(save, fh, indent=2)
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()
