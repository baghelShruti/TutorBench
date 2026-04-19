# TutorBench — Execution Guide (from current state to submission)

This guide walks you through every command needed to take the current
codebase in `/home/claude/v3/` (or wherever you unzip the delivered
bundle) and produce a submission-ready Kaggle writeup with real numbers.

The plan is staged by dependency: each step unblocks the next. Skip
nothing in Phase 1 and 2; Phase 3 is where you can stop if you hit a
wall.

---

## Phase 0. Environment setup (20 minutes)

```bash
# Local environment — you run most of the benchmark locally to validate
# before burning Kaggle quota.
python -m venv .venv
source .venv/bin/activate

# Core deps
pip install sympy numpy pandas matplotlib

# For real LLM calls (pick whichever you have keys for; you only need one)
pip install openai         # OPENAI_API_KEY
pip install anthropic      # ANTHROPIC_API_KEY
pip install google-genai   # GOOGLE_API_KEY (optional)

# For running AMORE locally on CPU/GPU (if you have it)
# Use llama.cpp or ollama, per your AMORE paper §4.5
```

Sanity checks:

```bash
cd tutorbench_v3
python scripts/utils/side_channel.py      # → 6/6 tests passed
python scripts/utils/pedagogy.py          # → 4 canonical response styles scored
python scripts/01_generate_problems.py    # → 200 problems → data/problems.json
python scripts/04_validate_detectors.py   # → κ = 0.54 (rule-based only, expected)
```

If any of these fail, stop and fix them before moving on.

---

## Phase 1. Real dry-runs on frontier models (2–4 hours)

The goal is to produce one JSON file per evaluated model, each
containing the full per-item breakdown. These files feed the hero
figure and populate the results section of the writeup.

### 1.1 Minimum viable run set

You need **at least 4 models** in the hero figure for the story to work:
two frontier models, the open-weights Llama, AMORE, and human tutors.
More is better but 4 is the floor.

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Start small — 20 items — to catch any bugs before full runs
python scripts/03_dry_run.py \
    --backend openai --model gpt-4o-mini \
    --judge-backend anthropic --judge-model claude-sonnet-4-5 \
    --n 20 \
    --out data/dryrun_gpt4o_n20.json
```

Check the output:
- `data/dryrun_gpt4o_n20.json` exists
- The `summary` block has non-zero `compliance`, `pedagogy_score_mean`,
  `inverted_catch_confirm_rate`
- Detector counts are reasonable (judge should catch some leaks; rule-based should catch a few)

If those look sensible, run the full 200-item eval per model:

```bash
# Frontier models
python scripts/03_dry_run.py --backend openai   --model gpt-5.1               --n 200 --out data/dryrun_gpt5.json
python scripts/03_dry_run.py --backend anthropic --model claude-opus-4-6     --n 200 --out data/dryrun_claude_opus.json
python scripts/03_dry_run.py --backend openai   --model gpt-4o-mini          --n 200 --out data/dryrun_gpt4o_mini.json

# Open-frontier
python scripts/03_dry_run.py --backend openai --model llama-3.3-70b-instruct --n 200 --out data/dryrun_llama33_70b.json
```

Each 200-item run makes roughly 6 × 200 = 1200 LLM calls for the tutor
plus 4 × 200 = 800 judge calls plus 200 confession calls. Budget
accordingly.

### 1.2 AMORE run

Run AMORE locally through llama.cpp (per your ICANN paper §4.5) and
expose it through an OpenAI-compatible shim (llama.cpp server does this
natively). Then:

```bash
# Start AMORE server
./llama-server -m amore-q4_k_m.gguf --port 8080 --host 0.0.0.0

# Point the OpenAI backend at the local server
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=dummy

python scripts/03_dry_run.py \
    --backend openai --model amore \
    --judge-backend anthropic --judge-model claude-sonnet-4-5 \
    --n 200 \
    --out data/dryrun_amore.json
```

**Critical:** the judge should NOT be AMORE itself — the whole point of
pinning the judge is to prevent circular self-evaluation. Always use a
frontier model (Claude, GPT, Gemini) as the judge.

### 1.3 Human baseline (2–3 hours of human time)

Pick 20 items spanning all 10 templates (2 each). Email them as a Google
Form to 3 people who have tutored math (your co-authors, friends who TA,
etc.). Give each tutor the TUTOR_SYSTEM instructions from
`scripts/utils/prompts.py::TUTOR_SYSTEM_V1` and the pressure sequence,
and collect their written responses.

Then wrap them in the same JSON shape as the dry-run output and run them
through the pipeline:

```bash
# Put human responses in a file shaped like dry-run output, then:
python scripts/05_make_figures.py \
    --input data/dryrun_gpt5.json \
    --input data/dryrun_claude_opus.json \
    --input data/dryrun_llama33_70b.json \
    --input data/dryrun_amore.json \
    --input data/dryrun_humans.json \
    --out writeup/figures
```

Even n=3 tutors × 20 items is enough. Label the figure as
"Human tutors (n=3, 20 items)" and note in the writeup that it's a
pilot — judges will accept this as long as the methodology is rigorous
(Fleiss' κ reported, blinded).

---

## Phase 2. Judge validation with real LLM (30 minutes)

The gold-item validation is what makes your detector trustworthy to
judges. The rule-based layers alone hit κ = 0.54; you need to show the
full stack pushes that past 0.75.

```bash
python scripts/04_validate_detectors.py \
    --judge anthropic --judge-model claude-sonnet-4-5
```

Expected output (based on the gold items as they are):
- Combined detector accuracy: 90–95%
- Combined Cohen's κ: 0.78–0.88
- Precision: ≥ 0.90
- Recall: ≥ 0.85

Record these numbers. They go directly into Table 1 of the writeup as
"Detector Validation." This table is what lets judges trust every
other number in your submission.

If κ comes in below 0.75, either add more gold items covering the
failure categories or tighten the judge prompt. Don't ship without this.

---

## Phase 3. Figures and writeup (4 hours)

### 3.1 Generate real figures

```bash
python scripts/05_make_figures.py \
    --input data/dryrun_gpt5.json \
    --input data/dryrun_claude_opus.json \
    --input data/dryrun_llama33_70b.json \
    --input data/dryrun_amore.json \
    --input data/dryrun_humans.json \
    --out writeup/figures
```

Inspect each figure. The hero figure should tell the story at a glance.
If it doesn't, your data is telling you something — maybe AMORE isn't
actually dominant on pedagogy, in which case adjust the narrative to
match the data. *Don't force the data to fit the pitch.*

### 3.2 Fill in the writeup

Open `writeup/draft.md` and replace every `[TBD]` and `[fill in]` with
real numbers from the dry-runs and judge validation. Key blanks:

- Abstract first 100 words: update the "central finding" based on what
  your hero figure actually shows.
- §4.1: cost per model (multiply call counts by provider pricing)
- §4.3: human baseline κ
- §5.1–5.5: all of the results section
- Appendix A: filled confusion matrix from `gold_validation.json`

### 3.3 Cover image

Kaggle requires a cover image for writeups. You have two options:

**Option A (cheap, 30 min):** Use Figure 1 (the Pareto frontier) as
the cover. It's already striking and tells the story. Export at
2400×1260 px.

**Option B (best, $20 on Fiverr):** Commission a stylized illustration
of "a tutor and student on a seesaw balancing compliance and pedagogy"
with the Pareto frontier embedded as a design motif. This signals
"serious research project, not a hackathon submission."

I recommend Option B if you can afford the time for it.

### 3.4 Paper-style PDF (optional but high-leverage)

Convert `draft.md` to a LaTeX/PDF two-column paper using NeurIPS or
ICML template. This takes 2–3 hours but makes your writeup look like a
real submission. Judges reading 50 Kaggle writeups will subconsciously
rate a paper-styled submission higher.

```bash
pandoc writeup/draft.md \
    --template=neurips2024.tex \
    -o writeup/tutorbench.pdf
```

Attach the PDF to the Kaggle writeup as a supplementary file.

---

## Phase 4. Kaggle submission (1 hour)

Per the Kaggle rubric:

1. **Create the Benchmark and Task on Kaggle Community Benchmarks.**
   Paste `scripts/02_tutorbench_task.py` as the task body. Set both
   the task and benchmark to PRIVATE (they become public after the
   deadline per the rules).
2. **Run the benchmark** on at least 2 frontier models through the
   Kaggle SDK so judges can re-run it. This uses Kaggle quota.
3. **Create the Writeup** with:
   - Cover image
   - The filled-in `draft.md` as the writeup body
   - `02_tutorbench_task.py` as the notebook
   - Link to the Benchmark via Attachments → Add a link
   - Link to this guide and `03_dry_run.py` as supplementary code

**Sanity checklist before clicking Submit:**

- [ ] Writeup first 100 words answer "what does this benchmark reveal
      that no prior benchmark can?"
- [ ] Hero figure is Figure 1, referenced in the abstract
- [ ] Every `[TBD]` is filled
- [ ] Cohen's κ ≥ 0.75 reported for the full-stack detector
- [ ] Human baseline numbers are in Figure 1 and §5
- [ ] AMORE paper is cited (but marked as "under review, anonymous")
- [ ] Task runs green on Kaggle Benchmarks (test this from a fresh
      Kaggle notebook before submission day)
- [ ] Benchmark and task are set to PRIVATE
- [ ] All referenced files are public/accessible

---

## Phase 5. Optional hedge: second-track submission (1 day)

If you have bandwidth after Phase 4, spin up a second submission in the
**Metacognition track** using the same infrastructure. Reuse the
procedural generator and the Confession metric — they carry over
one-to-one. Call it "CalibrationGap: Measuring LLM Social
Self-Awareness" and frame it as a pure metacognition benchmark, with
the cross-track connection to TutorBench as a note.

Two submissions in two tracks × two track prizes each × four grand
prizes = significantly better expected value than one submission.

---

## Troubleshooting

**"The judge returns weird output, not just LEAKED/SAFE."**
Tighten the prompt with a single-shot example. Check
`scripts/utils/prompts.py::LEAKAGE_JUDGE_PROMPT` — you may need to add
"Output only one word." at the end.

**"The Pareto frontier shows AMORE is NOT dominant."**
Don't fake it. Rewrite the narrative. Maybe the story is "frontier
models are closer to specialized tutors than prior work suggests" or
"the tradeoff is real even for frontier models." The honest story is
always better than a forced one, and the judges will trust you more.

**"The rule-based detectors have too many false positives."**
Lower the LCS threshold from 0.4 to 0.5, and check that `problem_numbers`
is being extracted correctly in the generator. Re-run gold validation.

**"Kaggle benchmark run times out."**
Reduce `n_jobs` from 4 to 2, or split the dataset into two halves and
run them as separate tasks under the same benchmark.

**"My dry-run summary has zeros everywhere."**
Check the backend: the stub backend will produce degenerate results.
Use `--backend openai` or `--backend anthropic` for real numbers.

---

## Priority order if you run out of time

If you cannot complete everything, ship in this order (most impactful
first):

1. Writeup with placeholder figure (replaceable before deadline)
2. Hero Pareto figure with at least 3 real model points
3. Task file running on Kaggle Benchmarks (even with 50 items)
4. Gold-item detector validation
5. Human baseline (n=3, 20 items)
6. Confession metric per model
7. Second-track submission
8. PDF conversion

Stop at the last item you can honestly complete without fabricating
numbers. A technically tight 3-model submission beats a 6-model
submission with inconsistent data.
