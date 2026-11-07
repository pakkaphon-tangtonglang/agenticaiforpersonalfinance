# Multi-intent routing: evaluation (2026-11-07)

Branch `feat/multi-intent-routing`. The router may split one message into one task per agent
(no cap; the bound is the 6 agents). Only the router changed, so only the router and the
multi-topic path were re-tested; single-topic answers run the same code as before.

## Run 1: single-topic routing regression (gemini-3.5-flash, minimal thinking, 185 questions, 1 round)
Results: `data/evaluation/results/router_thinking/multi_intent_prompt/report.md`

| | Old prompt (`prompt_v2`) | Multi-intent prompt |
|---|---|---|
| Accuracy | 177/185 = 95.7% | 175/185 = 94.6% (95% CI 90.3–97.0%) |

- −1.1 points, not significant: paired McNemar p = 0.69 (4 right→wrong, 2 wrong→right). The
  same size of change as between rounds of the old prompt (p = 0.45 and 0.75 vs the two
  `main` rounds). Passes the gate set before the run (within about 1.5 points).
- Wrong splits: 1/185. `s35_route_ambig_005` "ช่วยดูภาพรวมการเงินแล้วแนะนำด้วย" ("look at my finances overall, then advise me") went to
  report + recommendation. It does ask for two things and the user gets both answers, but the
  strict rule (a one-topic question must go to exactly one agent) counts it as wrong.
- Latency unchanged: 1.38 s median model call; input tokens about 1,854 per call.

## Run 2: multi-topic routing (same setup, 30 new questions, 1 round)
Data: `data/evaluation/routing_multi_dataset.yaml`. Results: `.../router_thinking/multi_topic/report.md`

- Multi-topic messages (12 with 2 topics, 6 with 3, 3 with 4, 1 with 5): exact agent set
  **22/22** (95% CI 85.1–100%), mean Jaccard 1.000, same order as asked 22/22.
- One question that sounds like several (e.g. "ซื้อประกันชีวิต 50,000 จะลดภาษีได้เท่าไร", "how much tax do I save if I buy 50,000 of life insurance";
  greeting + question; two expenses in one message): **8/8 correct, 0 wrongly split**.
- Limits: 30 questions written by us, one round; the CI lower bound (85%) is the honest claim.

## Run 3: Study A, hub_multi on the cross-topic tasks hub_spoke failed (DeepSeek V4.1 Flash, 3 rounds)
Results: `.../architecture_v2/ollama_deepseek-v4.1-flash_hub_multi/` (`report.md`, `merged.md`)

The six cross-topic tasks with any hub_spoke failure (crs_03, 05, 07, 08, 09, 12) were re-run
with `hub_multi` in all three rounds (18 runs). Every other hub run keeps its audited
hub_spoke result.

| Design | All 60 tasks | Cross-topic (12 × 3) | Cross pass^3 |
|---|---|---|---|
| hub_spoke (audited) | 157/180 = 87.2% | 22/36 = 61.1% | 6/12 |
| **hub_multi (merged, estimate)** | **166/180 = 92.2%** | **31/36 = 86.1%** | **10/12** |
| single_agent (audited) | 178/180 = 98.9% | 36/36 = 100% | 12/12 |

- Paired on the 18 re-run pairs: hub_spoke only 1, hub_multi only 10, exact McNemar
  **p = 0.012**.
- Cost on those tasks: LLM calls 3.4 → 5.8, input tokens 11.2k → 24.1k, median latency
  9.0 → 13.0 s. One-topic messages are unaffected.
- All 5 remaining failures were read by hand; none overridden:
  - crs_05 r0, r1: tax correct (36,500), but the router sent "ควรออมเพิ่มเดือนละเท่าไร" ("how much more should I save each month") to
    recommendation instead of planning; the answer gives a guideline figure (20% of income,
    12,000/month, cited) instead of the saving-plan calculator. Borderline, kept as a failure.
  - crs_12 r0–r2: expense part correct (300 baht); the tax section was **empty** (the known
    DeepSeek tax-agent empty reply). This exposed a product bug: an empty agent reply showed an
    empty heading. Fixed: an empty part now shows "ส่วนนี้ตอบไม่สำเร็จ…" ("this part could not be answered…") (tests added).

**Why the merged number is an estimate:** the six cross-topic tasks that never failed were not
re-run with hub_multi. Run 1 shows single-topic routing did not change significantly, and
run 2 shows multi-topic splitting is accurate, so their results are unlikely to drop, but this
was not measured. A full fair claim would re-run all 12 cross-topic tasks (36 runs).

## Thesis reading
- Multi-intent routing closes most of the gap to the single agent on cross-topic tasks
  (61% → 86%, p = 0.012) while keeping hub-and-spoke's token advantage on one-topic messages.
- The single agent is still higher (100% cross-topic); the remaining hub_multi failures are a
  routing choice (saving question → recommendation) and DeepSeek's empty tax replies.
- ch.3 must change if this ships: the router no longer always "ส่งให้ Agent เฉพาะทาง 1 ตัว" ("sends to one specialist agent").
