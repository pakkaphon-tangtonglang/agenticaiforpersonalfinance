# Evaluation methods: what published studies do (research notes, 2026-11-02)

Purpose: design the three comparison studies (architecture, model, embedding) the way
published work does, so the thesis (ch.4) and the portfolio report hold up to review.

## 1. Architecture comparisons

**Kim et al. (Google Research / DeepMind / MIT), "Towards a Science of Scaling Agent
Systems", arXiv 2512.08296 (Dec 2025).**
- Compares a single-agent baseline with four multi-agent designs: Independent,
  Centralized (orchestrator), Decentralized (peer-to-peer) and Hybrid, across several
  benchmarks and three LLM families.
- Key measure besides success rate: **error amplification**, i.e. how much one agent's
  mistake spreads. Independent agents amplified errors 17.2x; centralized ones contained it
  to about 4.4x because the orchestrator acts as a checkpoint.
- Finding: more agents is not automatically better; it is a trade-off between parallel work
  and coordination overhead.

**Gao et al., "Single-agent or Multi-agent Systems? Why Not Both?", arXiv 2505.18286 (May 2025).**
- The accuracy edge of multi-agent systems shrinks as the base LLM gets stronger.
- Reports accuracy and deployment cost side by side; proposes routing between the two.

**Copy into our study:** the same tasks, tools, prompts and model for every architecture; a
real single-agent baseline (one agent, all tools); measure success, LLM calls, tokens,
latency, and error propagation (a wrong route that leads to a wrong answer). Repeat on a
second model to test Gao's claim that the multi-agent edge shrinks with a stronger model.

## 2. Model comparisons

**Artificial Analysis (artificialanalysis.ai):** reports quality against **cost per task**
(not per token) and draws a **Pareto frontier**; latency is time to first answer token
*including thinking time*, plus output speed and end-to-end time.

**Yao et al., τ-bench, arXiv 2406.12045 (ICLR 2025):** **pass^k**, the chance that *all* k
repeated runs of a task succeed. Shows reliability falls sharply as k grows (gpt-4o retail:
~61% pass^1 → ~25% pass^8). pass^1 stays the main number.

**BFCL (Berkeley Function-Calling Leaderboard, ICML 2025):** checks tool calls structurally
(right function, right arguments, AST match), separately from the final answer.

**Copy into our study:** cost per question in THB from token counts; quality-vs-cost Pareto
chart; first-token and total latency; pass^3 consistency; tool-call correctness scored
separately from answer correctness.

## 3. Embedding comparisons

**MTEB / MMTEB (arXiv 2502.13595), MIRACL-th, NanoMTEB-Thai:** retrieval scored with
**nDCG@10** and **Recall@k** on a fixed corpus, always next to a **BM25** keyword baseline.
On NanoMTEB-Thai miracl_th, BM25 reaches nDCG@10 0.60 against 0.81 for a small dense model.

**Thai MTEB community leaderboard (github.com/anusoft/thai-mteb-leaderboard), SEA-BED
(arXiv 2508.12243):** strong Thai candidates are Qwen3-Embedding, multilingual-e5-large-instruct
and bge-m3; no peer-reviewed study settles bge-m3 vs mE5 for Thai.

**Copy into our study:** BM25 baseline; nDCG@10, Recall@5, MRR; same chunks for every model;
also check that better retrieval gives better final answers (end-to-end), plus index build
time, query latency and cost.

## 4. Statistics

**Miller (Anthropic), "Adding Error Bars to Evals", arXiv 2411.00640 (Nov 2024).**
- Compare systems on **question-level paired differences**, not two separate averages
  (smaller standard error when both are run on the same questions).
- Use **clustered standard errors** when several questions share one source (e.g. several
  RAG questions about the same knowledge-base file).
- Report 95% CIs; do a **power analysis** to choose how many questions are needed.

**Copy into our study:** paired tests, clustered SE by source document, power analysis
to size each dataset before running.

## 5. LLM judges

**RAGAS (Es et al., arXiv 2309.15217):** faithfulness = split the answer into claims and
check each against the retrieved context; it measures consistency with the context, not truth.

**Judge-validation practice (Ragas judge alignment docs; Evidently; "Judging the Judges",
arXiv 2406.12624):** hand-label a sample, report the judge's agreement with humans
(true-positive / true-negative rates or Cohen's κ), use binary pass/fail per failure mode,
use a judge from a different model family, and don't trust high agreement blindly.

**Copy into our study:** every judged metric gets a hand-labelled check set and a reported
agreement number; judge model differs from the model under test.

## 6. Error analysis

Practitioner guidance (Ragas, Evidently, hamelsmu evals-skills): read real failures first,
group them into a failure list, then build one check per failure. A failure breakdown table
is often more useful to readers than the headline accuracy.

## Sources
- https://arxiv.org/html/2512.08296v2 · https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/
- https://arxiv.org/abs/2505.18286v1
- https://artificialanalysis.ai/models
- https://arxiv.org/pdf/2406.12045 · https://sierra.ai/blog/benchmarking-ai-agents
- https://gorilla.cs.berkeley.edu/leaderboard.html · https://proceedings.mlr.press/v267/patil25a.html
- https://arxiv.org/pdf/2411.00640 · https://anthropic.com/research/statistical-approach-to-model-evals
- https://arxiv.org/pdf/2309.15217
- https://docs.ragas.io/en/latest/howtos/applications/align-llm-as-judge/ · https://www.evidentlyai.com/blog/how-to-align-llm-judge-with-human-labels · https://arxiv.org/html/2406.12624
- https://web3.arxiv.org/abs/2502.13595v2 · https://arxiv.org/pdf/2508.12243 · https://github.com/anusoft/thai-mteb-leaderboard
- https://hakari-bench-leaderboard.hf.space/docs/benchmark-tasks/NanoMTEB-Thai/miracl_th
