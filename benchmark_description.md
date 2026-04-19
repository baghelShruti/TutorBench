# TutorBench: Compliance × Pedagogy Frontier

When an LLM tutor is pressured by a student to reveal the answer, it faces a genuine conflict: refusing is rude; complying undermines learning. Existing benchmarks measure one axis or the other — none reveal the tradeoff. **TutorBench** locates frontier LLMs on a 2-D Pareto frontier between *compliance-under-pressure* and *pedagogical scaffolding quality*, using 200 procedurally-generated, SymPy-verified math problems no model has seen in training.

## What This Benchmark Measures

**Pedagogical Integrity Index (PII)** — a weighted harmonic mean of three dimensions:

1. **Compliance** — Can the model resist revealing the answer across a 4-level escalating pressure gradient? (direct request → social engineering → emotional manipulation → authority + confirmation)
2. **Inverted-Catch Confirmation** — When a student presents the *correct* answer, does the tutor appropriately confirm it? (Decorrelates from generic refusal)
3. **Pedagogy Quality** — Does the model use Socratic questioning, scaffolding, and hints without spoiling?

## Key Findings (6 frontier models)

| Model | PII Score |
|---|---|
| GPT-5.4 mini | 0.40 |
| Gemini 2.5 Flash | 0.36 |
| Claude Sonnet 4 | 0.32 |
| DeepSeek-R1 | 0.23 |
| Claude Sonnet 4.5 | 0.15 |
| Gemini 2.5 Pro | 0.15 |

**Surprising finding: more capable models make worse tutors.** Larger models are more eager to be "helpful" and cave under social pressure. The ranking is nearly inverse to general capability.

## Novel Contributions

- **Four-layer leakage detector** catching side-channel leaks that prior LCS-based detectors miss (AMORE's LCS metric catches 0/600 leaks at published thresholds)
- **Confirmation-driven leakage** — a distinct failure mode from classical goal decay, where models conceal at 97%+ under answer-free pressure but collapse to 20% when students assert an answer
- **Confession gap metric** — models systematically under-report their own leakage (calibration gap of −0.61), linking social cognition to metacognition
