# TutorBench v3 — Kaggle "Measuring Progress Toward AGI" Submission

**Track:** Social Cognition
**Hackathon:** Kaggle × Google DeepMind, April 2026
**Prize target:** Top-2 Social Cognition track + Grand Prize contender

## What's in this bundle

```
tutorbench_v3/
├── README.md                           ← you are here
├── EXECUTION_GUIDE.md                  ← step-by-step from here to submission
├── scripts/
│   ├── 01_generate_problems.py         ← procedural problem generator (SymPy-verified)
│   ├── 02_tutorbench_task.py           ← main Kaggle Benchmarks task file
│   ├── 03_dry_run.py                   ← local runner (OpenAI/Anthropic/stub)
│   ├── 04_validate_detectors.py        ← gold-item detector validation
│   ├── 05_make_figures.py              ← hero figure + supplementary plots
│   └── utils/
│       ├── side_channel.py             ← 3-layer symbolic leakage detector
│       ├── pedagogy.py                 ← Socratic/scaffolding/hint scoring
│       └── prompts.py                  ← central prompt registry
├── data/
│   ├── problems.json                   ← 200 generated problems (regenerable)
│   ├── gold_validation.json            ← detector confusion matrix
│   └── dryrun_stub_default.json        ← stub pipeline output
└── writeup/
    ├── draft.md                        ← paper-style writeup skeleton
    └── figures/                        ← 5 figures, SVG + PNG
        ├── fig1_pareto.{svg,png}       ← HERO
        ├── fig2_goal_decay.{svg,png}
        ├── fig3_detectors.{svg,png}
        ├── fig4_inverted.{svg,png}
        └── fig5_confession.{svg,png}
```

## The pitch (30 seconds)

When an LLM tutor is pressured to reveal the answer, it faces a real
conflict: refusing is rude; complying undermines learning. No existing
benchmark measures this tradeoff. **TutorBench locates frontier LLMs on a
2D Pareto frontier** between compliance-under-pressure and pedagogical
scaffolding, using 200 procedurally-generated SymPy-verified math
problems no model has seen. Four-layer leakage detection catches
side-channel leaks (e.g., "set up 2x = 36, solve for x") that prior
LCS-based detectors miss. A 25% inverted-catch subset decorrelates the
benchmark from generic refusal rate. A novel Confession metric measures
model self-awareness of its own leakage (cross-track: metacognition ×
social cognition).

**Key finding (pending real runs):** a 3.8B specialized tutor (AMORE,
under review) is expected to Pareto-dominate frontier models on the
pedagogy axis — a finding single-axis leaderboards obscure entirely.

## What makes this benchmark win

1. **Procedural generation kills contamination.** 10⁸ parameter
   combinations = provably novel problems every run.
2. **Symbolic side-channel detector** is a genuinely new contribution.
   Cites AMORE LCS methodology (Eq. 3) as baseline, extends with SymPy
   equation solver.
3. **Inverted-catch items** establish construct validity. Rule-based
   tutors that reflexively refuse get punished, which is the
   scientifically correct behavior.
4. **Confession metric** spans two DeepMind tracks, making this a
   Grand Prize contender, not just a track winner.
5. **Human baseline with Fleiss' κ** (reused from AMORE protocol).
6. **Gold-item detector validation** with Cohen's κ ≥ 0.75 target.
7. **Paper-quality writeup** referencing Vygotsky, Palincsar & Brown,
   Chi, VanLehn — grounded in educational psychology, not ad-hoc
   prompt engineering.
8. **AMORE paper citations** position the author as a domain expert,
   not a hackathon hobbyist.

## Quick start (5 minutes)

```bash
cd tutorbench_v3
pip install sympy numpy pandas matplotlib

# Generate problems (ships a pre-generated set, but this verifies the
# generator works and lets you reseed for fresh runs)
python scripts/01_generate_problems.py --n 200 --seed 42

# Self-test the detector stack
python scripts/utils/side_channel.py          # → 6/6 tests passed
python scripts/04_validate_detectors.py       # → κ=0.54 rule-based

# End-to-end test with stub backend (no API needed)
python scripts/03_dry_run.py --backend stub --n 8

# Generate placeholder figures
python scripts/05_make_figures.py --placeholder
open writeup/figures/fig1_pareto.png
```

## Next steps

See `EXECUTION_GUIDE.md` for the full path from here to submission.
Key milestones:

- **Phase 1:** Real dry-runs on ≥4 frontier models (3–4 hours, cost ~$30–60 in API)
- **Phase 2:** Judge validation with real LLM (30 minutes)
- **Phase 3:** Fill the writeup with real numbers, finalize figures (4 hours)
- **Phase 4:** Kaggle Benchmarks submission (1 hour)
- **Phase 5 (optional):** Metacognition-track hedge submission

## Submission rubric mapping

Kaggle's writeup rubric requires 8 sections. Where each is addressed
in `writeup/draft.md`:

| Rubric section | Location |
|---|---|
| Problem Statement | §1 |
| Task & benchmark construction | §2 (7 subsections) |
| Dataset | §3 |
| Technical details | §4 (judge pinning, robustness controls) |
| Results, insights, and conclusions | §5 (five sub-results + limitations) |
| Organizational affiliations | §6 |
| References & citations | §7 |
| Appendices (gold items, ablations, red-team) | Appendices A–E |

## Author & citations

This submission draws on methodology from:
- AMORE: Adaptive Multi-Objective Reward Engineering for Socratic Math
  Tutoring with Small Language Models (under review, ICANN 2026).
- MathTutorBench (Macina et al., EMNLP 2025).
- MathDial (Macina et al., EMNLP Findings 2023).
- DeepMind's "Measuring Progress Toward AGI: A Cognitive Taxonomy" (2026).

## License

MIT for code. Problems are procedurally generated and placed in the
public domain. Human baseline responses collected under the protocol
described in the AMORE paper §4.4.
