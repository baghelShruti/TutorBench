What I Built
The script (generate_human_baseline_form.py) generates N self-contained HTML forms, each with:

20 items covering all 4 pressure levels (stratified: 5 per level)
5 unique problems per form, different subsets across forms for coverage
Full conversation context matching exactly what LLMs saw
Embedded JavaScript for auto-save to browser session storage
"Download Responses" button exports JSON matching your pipeline format

The 5 ready-to-send forms (form_001.html through form_005.html) — each ~80KB, completely self-contained, work offline in any browser.
The example response file (EXAMPLE_RESPONSE.json) — shows exactly what structure the scorer will ingest.
Step-by-Step Instructions
Step 1: Find 3-5 people to recruit right now.
Friends who tutor, CS/math TAs, teachers, anyone who has tutoring experience. Text them:

"Hey — I'm running a research study on AI vs human math tutoring for a DeepMind/Kaggle hackathon. Need ~30 minutes of your time. You open an HTML file, respond to 20 math tutoring prompts as the tutor, click download, and send me the file. Deadline is tomorrow. Can you help?"

Step 2: Send each person a DIFFERENT form file. So form_001.html to person 1, form_002.html to person 2, etc. This maximizes problem coverage. Send via WhatsApp, email, Google Drive — any method.
Step 3: Tell them the process:

Download the HTML file
Double-click to open it in any browser (Chrome, Firefox, Safari — all work)
Fill it out — takes 25-40 minutes, can save progress and come back
Click "Download My Responses" at the bottom
Send back the downloaded JSON file

Step 4: Collect the returned JSON files into a folder:
/hackathon/human_baseline/responses/
├── human_baseline_form_001_alice.json
├── human_baseline_form_002_bob.json
├── human_baseline_form_003_carol.json
...
Critical Honest Notes
On sample size: 3-5 humans × 20 items = 60-100 data points. This is "preliminary reference", NOT a demographically representative baseline. The writeup must say this. DeepMind's framework calls for representative samples — we can't meet that in 24hrs, but we can be honest.
On expertise: Ideally "expert math tutors" but realistically "people with tutoring experience." The form records their background so you can report this honestly in the writeup.
What this gives you that no other team will have:

Direct human comparison on the exact same items LLMs saw
Human Confession Gap (metacognitive calibration for humans) — a world-first measurement
A figure showing human per-level concealment vs LLM per-level concealment — if humans don't show the L2/L4 zigzag but LLMs do, that's structural evidence for confirmation-driven leakage being AI-specific

What's Next
Once you have at least 3 responses back, I'll give you score_human_baseline.py which:

Ingests all the returned JSON files
Runs each human response through the same 4-layer leakage detector
Computes per-human, per-pressure-level concealment rates
Computes per-human Confession Gap (their self-report vs pipeline ground-truth)
Outputs a summary JSON comparable to your LLM results
Generates the comparison figures

Start the recruitment NOW — this is the critical path. While waiting for responses, we can do the arXiv upload and start the multi-model per-level runs in parallel.
Ready for me to build the scorer script, or do you want to recruit first and come back once you have at least one test response? If recruiting, I'd wait — I need to see one real returned JSON to make sure the scorer handles edge cases correctly.