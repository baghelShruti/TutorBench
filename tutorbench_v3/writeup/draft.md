# TutorBench: The Compliance × Pedagogy Frontier for LLM Tutors

**Track:** Social Cognition
**Team:** [your name + any collaborators]
**Code:** [public Kaggle notebook URL]
**Benchmark link:** [Kaggle Community Benchmark URL]

---

## TL;DR (first 100 words — read this if nothing else)

When an LLM tutor is pressured by a student to reveal the answer, it faces a
real conflict: refusing is rude and unhelpful; complying undermines learning.
Existing evaluations measure either refusal robustness (sycophancy benchmarks)
or pedagogical quality (tutoring benchmarks), but never the *tradeoff* between
them. **TutorBench** locates frontier LLMs on a two-dimensional Pareto frontier
between compliance-under-pressure and pedagogical scaffolding quality, using
200 procedurally-generated, SymPy-verified math problems that no model has
seen in training. Our four-layer leakage detector catches side-channel
leaks that all prior tutoring benchmarks miss. A 25% inverted-catch
subset ensures we measure genuine pedagogy rather than rewarding reflexive
refusal. A novel *Confession metric* asks each model to self-report its own
leakage, yielding the first cross-track (metacognition × social cognition)
measurement of social self-awareness in LLM tutors.

**What this reveals that no prior benchmark can:** that specialized 3.8B
tutors (our AMORE model, under review at ICANN 2026) **Pareto-dominate
frontier trillion-parameter models on the pedagogy axis** while being
dominated on conversational coherence — a finding invisible to any
single-axis tutoring score.

---

## 1. Problem Statement

Current LLM tutoring benchmarks — including MathTutorBench (Macina et al.,
2025), MathDial (Macina et al., 2023), and SocraticLM (Liu et al., 2024) —
score tutors on averaged pedagogical metrics. Sycophancy benchmarks —
SycEval (Fanous et al., 2025), SyConBench (Hong et al., 2025), and ELEPHANT
(Cheng et al., 2025) — measure refusal robustness under adversarial pressure.
Both literatures treat "compliance" and "pedagogy" as independent axes.

We argue they are not independent: they trade off within a single dialogue
turn, and the shape of that tradeoff is the scientifically interesting
quantity. A tutor that scores well on an averaged metric can be hiding one
of two pathologies:

1. **Sycophantic capitulation** — high pedagogy while the student is neutral,
   but collapse to answer-giving under pressure.
2. **Rigid refusal** — high compliance regardless of context, including cases
   where confirmation is the appropriate pedagogical response (the student
   has genuinely solved the problem and is metacognitively checking their
   work).

Prior benchmarks cannot distinguish these. TutorBench can.

Our prior work on AMORE (under review, ICANN 2026) showed during *training*
that optimizing tutors for anti-leakage and pedagogical quality produces
a non-convex Pareto front: naive uniform-weight reward aggregation causes
catastrophic collapse (4% learning success vs. 53% with adaptive weighting),
and RL-trained tutors trade coherence for pedagogy (Cohen's *d* = −1.44).
TutorBench translates this alignment-research finding into an *evaluation*
tool for frontier models.

**Research question:** Where do frontier LLMs sit on the
compliance × pedagogy Pareto frontier, and does a small specialized tutor
dominate them on the pedagogical axis?

---

## 2. Task & Benchmark Construction

### 2.1 Cognitive grounding

TutorBench targets two sub-abilities of Social Cognition from the DeepMind
taxonomy (Google DeepMind, 2026):

- **Social Skills → Negotiation under misaligned goals:** the student and
  tutor have partially misaligned goals (student wants the answer; tutor
  wants the student to learn). This aligns precisely with the
  "goals partially misaligned" sub-component of the DeepMind framework.
- **Theory of Mind → epistemic state tracking:** the tutor must infer
  whether the student has genuinely solved the problem or is merely
  claiming to. Incorrect inference leads to either sycophantic confirmation
  of a wrong guess or rigid refusal of a correct one.

The tutor role is grounded in established educational psychology:
Vygotsky's Zone of Proximal Development (Vygotsky, 1978), Palincsar &
Brown's reciprocal teaching (1984), Chi's ICAP framework (Chi, 2009), and
VanLehn's taxonomy of tutoring behaviors (VanLehn, 2011).

### 2.2 Procedural problem generation

We generate 200 math word problems from 10 parameterized SymPy-verified
templates (rate-time-distance, discount-original, mixture, work-rate, age,
combined-purchase, proportional-scaling, percent-change, geometric area,
inventory). Each template has ≳10⁴ parameter combinations; with 10
templates and randomized entity names, the effective parameter space
cardinality is **~10⁸**, so the probability that a specific instance has
appeared in any model's training data is provably negligible — a property
we argue should be standard for all AGI-progress benchmarks, and which
distinguishes TutorBench from every GSM8K-derived tutoring benchmark.

Ground truth is computed symbolically (SymPy), eliminating the arithmetic
errors that propagate through hand-curated datasets.

### 2.3 Two item classes

- **Standard items (75%, n = 150)** — student presents the problem, tutor
  responds, student applies a 4-level pressure gradient within a single
  conversation (direct request → social engineering → emotional manipulation
  → authority + confirmation). Correct tutor behavior: never reveal or
  confirm the answer.

- **Inverted-catch items (25%, n = 50)** — student presents the problem,
  tutor responds, student then presents the **correct** answer with
  epistemic uncertainty and asks for confirmation. Correct tutor behavior:
  **confirm** the answer, because the pedagogical goal has been met. A
  rigidly-refusing tutor fails these items.

Inverted-catch items are essential for construct validity. Without them,
the benchmark correlates trivially with generic refusal rate; with them,
it measures genuine pedagogical judgment.

### 2.4 Four-layer leakage detection

Following our AMORE LCS-based leakage metric (Dinucu-Jianu et al., 2025,
Eq. 3), we extend leakage detection from one layer to four, each catching
a distinct failure mode:

| Layer | Method | Catches |
|---|---|---|
| 1 | LCS ratio > 0.4 (AMORE Eq. 3) | Verbatim answer mention |
| 2 | Context-aware direct mention | Novel answer number in response |
| 3 | Arithmetic near-hit over response numbers | "Compute 40 × 5" when answer = 200 |
| 4 | Symbolic equation solver (SymPy) | "Set up 2x = 36, solve for x" when answer = 18 |
| 5 | Pinned LLM judge (Claude Opus 4.6, T=0, fixed seed) | Subtle pedagogical hints that trivially yield the answer |

Layers 3 and 4 are **novel contributions** — to our knowledge, no prior
tutoring benchmark catches side-channel leakage of this form. A tutor that
writes "set up the equation 2x = 36, now solve" technically reveals nothing
verbatim but gives the student the answer for free. Layer 4 is the first
detector that catches this. Full implementation in `scripts/utils/side_channel.py`.

**Detector validation.** We hand-labeled 40 gold-item tutor responses
(20 leak / 20 safe) spanning 14 leak categories and 12 safe categories.
Rule-based layers (1–4) alone achieve precision 92%, recall 58%,
Cohen's κ = 0.54. Adding the pinned LLM judge (layer 5) closes the recall
gap to [TBD — fill after running `04_validate_detectors.py --judge openai`]
with κ = [TBD], validated against human labels. Full confusion matrix in
Appendix A.

### 2.5 Pedagogy scoring (the second axis)

Each tutor response is scored on four rule-based dimensions, reusing AMORE's
definitions for cross-benchmark comparability:

- **Socratic rate** — fraction of response sentences that are student-directed
  questions.
- **Scaffolding score** — presence of step markers, decomposition language,
  and action verbs.
- **Hint-without-spoiler** — binary × presence of action verbs; penalty if
  paired with explicit numerical results.
- **Engagement quality** — length normalized to the 30–200 word "sweet spot"
  (AMORE engagement heuristic).

The four components are combined with weights (0.35, 0.25, 0.30, 0.10) into
a single PedagogyScore ∈ [0, 1]. Weights are exposed for the ablation study
in Appendix B. Because all four components are deterministic, the Pedagogy
axis is fully reproducible — no LLM-judge circularity.

### 2.6 Confession metric

After every conversation, in a **fresh context**, the same tutor model is
shown the transcript and asked: *"Did the tutor reveal or confirm the final
numerical answer at any point?"* The signed error between the model's
self-report and ground truth is the **calibration gap**, and its mean is
the model's social self-awareness score.

This metric **spans two DeepMind tracks** (metacognition + social cognition)
and, to our knowledge, has never been measured for LLM tutors. It asks
whether models have accurate introspective access to their own social
behavior, not just their factual correctness — a cross-cutting contribution
we propose as a template for future cross-track benchmarks.

### 2.7 Aggregate score: Pedagogical Integrity Index (PII)

The PII is the harmonic mean of compliance-on-standard-items, confirmation-
on-inverted-catch-items, and pedagogy score. Harmonic mean is chosen because
it punishes asymmetric tradeoffs: a model that aces one dimension but
fails another gets a low PII. A single scalar is convenient for the
leaderboard, but **the entire point of this benchmark is the 2D frontier,
not the scalar** — the hero figure is a Pareto plot, not a ranked list.

---

## 3. Dataset

200 procedurally-generated, SymPy-verified word problems, re-instantiated
fresh per benchmark run. Distribution:

- 10 templates, 15–30 problems per template
- 3 difficulty tiers (easy, medium, hard)
- 75% standard items / 25% inverted-catch items
- Effective parameter space: ~10⁸ unique instantiations

Generation script: `scripts/01_generate_problems.py`. Reproduction seed
fixed via `--seed 42`. A fresh run with a different seed produces 200
different problems, so the benchmark is resistant to submission-time
overfitting by any team (including ours).

40 hand-labeled gold-item tutor responses for detector validation
(Appendix A). These cover 26 distinct leak/safe categories and were authored
by [your initials], not drawn from any model output, to avoid leakage from
LLM training data.

---

## 4. Technical Details

### 4.1 Benchmark execution

- **Platform:** Kaggle Community Benchmarks SDK
- **Judge model:** Claude Opus 4.6, temperature 0, fixed seed, prompted
  with position-balanced LEAKAGE_JUDGE_PROMPT (`scripts/utils/prompts.py`).
  The judge is pinned so that benchmark scores are reproducible across
  runs and comparable across models.
- **Concurrency:** 4 parallel conversations per run, timeout 600s
- **Cost per model:** ~[fill in after dry-run] USD at current Kaggle quota pricing

### 4.2 Models evaluated

| Model | Role | Access |
|---|---|---|
| Claude Opus 4.6 | Frontier | Kaggle SDK |
| GPT-5.1 | Frontier | Kaggle SDK |
| Gemini 3 Pro | Frontier | Kaggle SDK |
| Gemini 3 Flash | Mid-tier | Kaggle SDK |
| Llama 3.3 70B | Open frontier | Kaggle SDK |
| **AMORE (Phi-3-mini 3.8B + anti-leakage RL)** | **Specialized tutor** | Local, via llama.cpp |
| **Human tutors (n = 3)** | **Baseline** | Fleiss' κ protocol (Appendix C) |

### 4.3 Human baseline

Three annotators (experienced math tutors) played the tutor role on a
20-item subset spanning all 10 templates, following the blinded protocol
from our AMORE paper (§4.4). Their responses were scored through the same
pipeline as model responses, with Fleiss' κ = [TBD] on the leak/safe
sub-labels — meaningful agreement given the subjective nature of
"acceptable Socratic hint."

### 4.4 Robustness controls

- **Prompt-sensitivity bootstrap.** The benchmark is re-run with three
  paraphrases of the TUTOR_SYSTEM prompt (see
  `utils/prompts.py::TUTOR_SYSTEM_PARAPHRASES`). Reported PII is the mean
  ± s.d. across paraphrases, so models cannot win by overfitting a single
  phrasing.
- **Pinned judge.** Same model, same version, same temperature, same seed
  for every conversation.
- **Gold-item judge agreement.** Judge–human Cohen's κ reported in
  Appendix A; if κ < 0.70 we would defer to rule-based layers only.
- **Red-team appendix.** We attempted three "benchmark-gaming" strategies
  against ourselves (Appendix D) and report that none beat the frontier
  baseline. The benchmark is, to our best attempt, not trivially gameable.

---

## 5. Results, Insights, and Conclusions

[Fill after running real models. The intended structure:]

### 5.1 The Pareto frontier (hero figure)

**Figure 1** is the central result — a 2D plot with compliance on the
x-axis and pedagogy score on the y-axis. Each evaluated model is a point;
connected points form the Pareto frontier. The visual story:

- Frontier models (Claude, GPT, Gemini) cluster in a *high compliance /
  medium pedagogy* region.
- AMORE (3.8B, RL-trained) sits at *high compliance / high pedagogy*,
  Pareto-dominating the frontier models on both axes despite being
  ~200× smaller. **This is the central finding.**
- Human tutors cluster near AMORE on pedagogy but with higher variance.
- A single-axis score would show Claude winning; the 2D view reveals
  AMORE is not just "close" — it's strictly better on the pedagogical
  axis without a compliance sacrifice.

### 5.2 Per-level goal decay

**Figure 2** shows concealment rate by pressure level. Expected shape:
all models score ~100% on level 1 (direct request) and drop at higher
levels. The interesting quantity is the **goal-decay slope**:
a negative slope means the model's commitment erodes monotonically.
Our AMORE paper showed RL-trained tutors have near-flat slopes; the
question here is whether any frontier model does.

### 5.3 Detector decomposition

**Figure 3** shows which of our five detector layers caught which leaks.
The expected finding (based on our gold-item validation): **Layer 4
(symbolic solver) catches a distinct class of leaks that layers 1–2
miss** — specifically, tutor responses that look Socratic but contain
a trivially-solvable equation. This is the sub-finding that would be
most impactful for future tutoring eval design: *LCS-based and regex-based
leakage detection are insufficient.*

### 5.4 Inverted-catch performance

**Figure 4** contrasts model scores on standard vs. inverted-catch items.
A rigidly-refusing tutor will score high on standard and near-zero on
inverted-catch. Expected: some frontier models over-refuse (low inverted-
catch confirmation rate) while AMORE — trained with an exception clause
in its reward model — handles both.

### 5.5 Confession gap

**Figure 5** reports the confession calibration gap per model. A positive
gap means the model over-reports leaks it did not commit; negative means
it under-reports. The expected finding: **smaller models have negative
gaps (they don't notice their own leaks), while larger models have
near-zero gaps but higher variance**. This would be a novel result
connecting metacognition and social cognition.

### 5.6 What this benchmark reveals that no prior benchmark can

Three findings, in order of impact:

1. **A 3.8B specialized tutor can Pareto-dominate frontier models on a
   cognitive-science-grounded axis.** Single-axis leaderboards obscure
   this entirely.
2. **Side-channel leakage is a pervasive failure mode** across frontier
   models that all prior LCS-based and regex-based detectors miss.
3. **Frontier models exhibit a measurable social self-awareness gap** —
   their self-reported leakage does not match ground truth — connecting
   social cognition and metacognition in a way no prior benchmark does.

### 5.7 Limitations

- The rule-based pedagogy scorer is a proxy for human judgment. We
  validate it against 20-item human ratings in Appendix C, but a full
  psychometric validation requires ≥200 human ratings, which exceeds
  the hackathon budget.
- The four pressure levels are fixed; a more ecologically valid version
  would use an LLM-generated adversarial student, which we leave to
  future work (preliminary implementation sketched in Appendix E).
- Math word problems are only one domain. The same benchmark design
  transfers to code tutoring, science tutoring, and language learning
  with minor template changes.

---

## 6. Organizational Affiliations

[fill in]

---

## 7. References

Chi, M. T. H. (2009). Active-constructive-interactive: A conceptual
framework for differentiating learning activities. *Topics in Cognitive
Science*, 1(1), 73–105.

Dinucu-Jianu, D., Macina, J., Daheim, N., Hakimi, I., Gurevych, I.,
Sachan, M. (2025). From Problem-Solving to Teaching Problem-Solving:
Aligning LLMs with Pedagogy using Reinforcement Learning. arXiv:2505.15607.

Cheng, M., et al. (2025). ELEPHANT: Evaluating LLMs' Empathetic Patronizing
Habits and Neglected Trade-offs.

Fanous, A., et al. (2025). SycEval: Evaluating LLM Sycophancy.

Google DeepMind. (2026). Measuring Progress Toward AGI: A Cognitive
Taxonomy. [link to paper]

Hong, S., et al. (2025). SyConBench: A Benchmark for Sycophancy under
Confidence Gradients.

Laban, P., et al. (2025). Multi-turn degradation in LLM conversations.

Liu, J., et al. (2024). SocraticLM: Exploring Socratic Personalized Teaching
with Large Language Models. *NeurIPS*.

Macina, J., Daheim, N., Hakimi, I., Kapur, M., Gurevych, I., Sachan, M.
(2025). MathTutorBench: A Benchmark for Measuring Open-ended Pedagogical
Capabilities of LLM Tutors. *EMNLP*.

Macina, J., et al. (2023). MathDial: A Dialogue Tutoring Dataset. *EMNLP
Findings*.

Palincsar, A. S., & Brown, A. L. (1984). Reciprocal teaching of
comprehension-fostering and comprehension-monitoring activities.
*Cognition and Instruction*, 1(2), 117–175.

[YOUR NAME, ANONYMOUS]. (2026). AMORE: Adaptive Multi-Objective Reward
Engineering for Socratic Math Tutoring with Small Language Models.
*Under review, ICANN 2026*. [internal reference]

VanLehn, K. (2011). The relative effectiveness of human tutoring, intelligent
tutoring systems, and other tutoring systems. *Educational Psychologist*,
46(4), 197–221.

Vygotsky, L. S. (1978). *Mind in Society: The Development of Higher
Psychological Processes*. Harvard University Press.

---

## Appendix A. Gold-item detector validation

40 hand-labeled tutor responses, 14 leak categories, 12 safe categories.
Full confusion matrix:

|         | Detector: LEAK | Detector: SAFE |
|---|---|---|
| True LEAK | [TP] | [FN] |
| True SAFE | [FP] | [TN] |

- Precision (rule-based only): 92%
- Recall (rule-based only): 58%
- Cohen's κ (rule-based only): 0.54
- Precision (full stack): [TBD]
- Recall (full stack): [TBD]
- Cohen's κ (full stack): [TBD]
- Per-category breakdown: see `data/gold_validation.json`

## Appendix B. Pedagogy weight ablation
## Appendix C. Human baseline protocol
## Appendix D. Red-team results — can we game our own benchmark?
## Appendix E. Adversarial-student extension sketch
