# Router thinking level: notes for the thesis (2026-11-07)

Use this when writing ch.4 (router study) and when checking ch.3 l.201 / l.449
("minimal thinking level for the router: faster, routing accuracy unchanged").

## What the levels do (evidence)

**Our measurement**: thinking probe, `data/evaluation/results/router_thinking/thinking_probe/probe.md`
(raw calls in `probe.jsonl`). It sent 10 routing questions through the production router prompt and
3 maths questions, once per model × level. Thinking tokens are `output_token_details.reasoning`
as reported by Gemini.

| Model | Level | Routing: thinking tokens (calls that thought) | Maths: thinking tokens | Routing latency median |
|---|---|---|---|---|
| gemini-3.5-flash | minimal | 0 (0/10) | 0 | 1.51 s |
| gemini-3.5-flash | low | 0 (0/10) | 821 | 1.45 s |
| gemini-3.5-flash | medium | 323 (10/10) | 1,821 | 3.18 s |
| gemini-3.5-flash | high | 469 (10/10) | 2,510 | 3.82 s |
| gemini-3.5-flash | no setting | 333 (10/10) | 2,320 | 3.14 s |
| gemini-3.5-flash-lite | minimal / low / no setting | 0 | 0 / 620 / 0 | ~1.0 s |
| gemini-3.5-flash-lite | medium | 279 (9/10) | 2,002 | 1.88 s |
| gemini-3.5-flash-lite | high | 568 (10/10) | 1,712 | 2.47 s |

What this shows:
- On routing, **minimal and low both spend 0 thinking tokens** on both models. Minimal vs low
  therefore cannot show a difference for the router.
- The setting does reach the model: on maths, low thinks (821 tokens) and minimal does not (0).
- **3.5-flash with no setting thinks like medium** (333 vs 323 tokens, about 3.1 s). This is how
  the router ran before the latency fix, so it is the correct baseline for the ch.3 claim.
- **flash-lite with no setting does not think** (its default is minimal).

**Google documentation** (retrieved 2026-11-07):
- "Use minimal thinking for fact retrieval or classification": Gemini API docs, *Thinking*,
  https://aistudio.google.com/docs/thinking. This supports using minimal for the router.
- Default levels: Gemini 3.5 Flash-Lite = minimal (*Thinking* model table); Gemini 3.5 Flash =
  medium ("Uses the same `medium` default thinking level as Gemini 3.5 Flash", *What's new in
  Gemini 3.6*, https://aistudio.google.com/docs/whats-new-gemini-3.6). Both match the probe.
- The docs **do not define** each level. Do not write that a level is a "ceiling" or a fixed
  budget. The Gemini 3 developer guide (https://ai.google.dev/gemini-api/docs/gemini-3) reportedly
  describes it as the maximum depth of reasoning, but it was not opened directly, so do not quote it.

## Suggested thesis text (Thai, ch.4)

> Gemini รุ่น 3 ขึ้นไปให้กำหนดระดับการคิด (Thinking Level) ได้ 4 ระดับ ได้แก่ minimal, low, medium และ high
> เอกสารของ Google แนะนำให้ใช้ระดับ minimal กับงานค้นข้อเท็จจริงหรืองานจำแนก (Classification)
> คณะผู้จัดทำวัดจำนวนโทเค็นการคิดที่ Gemini รายงานกลับมา พบว่าเมื่อจำแนกเจตนาของคำถาม ระดับ minimal และ low
> ใช้โทเค็นการคิด 0 โทเค็นทั้งใน gemini-3.5-flash และ gemini-3.5-flash-lite แต่กับโจทย์คณิตศาสตร์ ระดับ low
> ใช้โทเค็นการคิดเฉลี่ย 821 โทเค็น ขณะที่ระดับ minimal ใช้ 0 โทเค็น แสดงว่าค่าที่กำหนดถูกส่งไปถึงโมเดลจริง
> แต่งานจำแนกเจตนาง่ายพอที่โมเดลจะไม่คิดเลยในระดับ low ส่วนกรณีที่ไม่กำหนดระดับการคิด ซึ่งเป็นค่าที่
> Router Agent ใช้ก่อนการปรับปรุง gemini-3.5-flash ใช้โทเค็นการคิดเฉลี่ย 333 โทเค็นต่อคำถาม และใช้เวลาประมาณ
> 3.1 วินาที เทียบกับ 1.5 วินาทีที่ระดับ minimal คณะผู้จัดทำจึงทดสอบข้อความในบทที่ 3 ด้วยการเทียบระดับ minimal
> กับกรณีไม่กำหนดระดับการคิด

## English (portfolio report)

> Google recommends minimal thinking for classification. On routing questions, minimal and low both
> used 0 thinking tokens on both models; on maths, low used 821 and minimal 0, so the setting does
> reach the model. With no setting (how the router ran before the latency fix), 3.5-flash used ~333
> thinking tokens per question and took ~3.1 s vs ~1.5 s at minimal. The study therefore tests
> "minimal, no loss in accuracy" as minimal vs no setting, not minimal vs low.

## Study design that follows (4.11 B, decided 2026-11-07)

| Setup | Model | Thinking | Answers |
|---|---|---|---|
| A | 3.5-flash | minimal (production) | — |
| B | 3.5-flash | no setting (before the fix) | A vs B: the ch.3 claim |
| C | flash-lite | minimal | C vs A: could flash-lite replace the router? |
| D | flash-lite | high | C vs D: does thinking help the cheap model? |

The probe numbers come from 10 + 3 prompts per cell. After the full run, cite the full-run averages
(185 questions) for setups A–D, and keep the probe only for the low-level finding.

## Results: full run, 1 round (2026-11-07, 740 calls, 0 errors, 0 unreadable replies)

`data/evaluation/results/router_thinking/main/report.md`

| Setup | Accuracy (95% CI) | Latency median / p95 | Thinking tokens |
|---|---|---|---|
| A flash minimal (production) | 95.7% (91.7–97.8) | 1.40 / 1.65 s | 0 |
| B flash, no setting | 96.2% (92.4–98.2) | 2.66 / 4.06 s | 272 |
| C lite minimal | 94.1% (89.7–96.6) | 0.89 / 1.15 s | 0 |
| D lite high | 95.7% (91.7–97.8) | 1.93 / 3.03 s | 471 |

- **ch.3 claim (A vs B):** accuracy −0.5 points (95% CI −3.2 to +2.2, McNemar p = 1.0);
  **1.28 s faster** per question (significant). Write it as: "no significant accuracy
  difference, and about half the latency". Do not write "proven no loss": with 1 round the CI
  reaches 3.2 points below zero, just past the 3-point margin.
- **Flash-lite instead of flash (C vs A):** −1.6 points (−4.9 to +1.6, p = 0.51) and 0.49 s
  faster. The difference is not significant, but the CI is too wide to call it equal. Report this
  as future work, not as a change.
- **Thinking on flash-lite (C vs D):** high thinking was +1.6 points (not significant) and 1.04 s
  slower.
- **Audit:** every wrong call was read. Three labels gained a second accepted answer, because the
  router prompt's own rules support what the models chose: pln_08 and rec_13 also accept
  general_chat ("ออมเงิน, เคล็ดลับการเงิน" is listed under general), and vag_07 also accepts
  report ("สถานะการเงินทั้งหมด"). The September labels were not changed, to keep them comparable,
  but 2 of them are debatable (ambig_002 "ต้องวางแผนใหม่" → planning, ambig_003 → tax).
- **Real errors worth naming:** "สวัสดีครับ" got the ask-back menu in 3 of 4 setups.
  Recommendation-type questions often went to general on flash-lite (82.6%). Thinking made
  ast_11 (RMF fund search) go to tax.
- To narrow the CIs, run round 2 (another 740 calls, roughly the same cost as round 1).

## Results: 2 rounds (2026-11-07, 1,480 calls, 0 errors) — these replace the 1-round numbers above

| Setup | Accuracy (95% CI) | Latency median / p95 |
|---|---|---|
| A flash minimal (production) | 95.9% (93.4–97.5) | 1.36 / 1.95 s |
| B flash, no setting | 96.2% (93.7–97.7) | 2.59 / 4.23 s |
| C lite minimal | 93.8% (90.8–95.8) | 0.88 / 1.16 s |
| D lite high | 96.5% (94.1–97.9) | 1.93 / 3.24 s |

- **ch.3 claim (A vs B): supported.** −0.3 points (95% CI −2.2 to +1.6, McNemar p = 1.0), so the
  whole CI is above the −3-point margin; **1.26 s faster** per question (sign test p < 0.0001).
  ch.3 can say "ความแม่นยำไม่ลดลงอย่างมีนัยสำคัญ" ("no significant loss in accuracy") with these
  numbers.
- **Flash-lite instead of flash (C vs A): not adopted.** −2.2 points (−4.6 to 0.0, p = 0.096),
  mostly on recommendation questions; 0.46 s faster. D (lite high) is as accurate (96.5%) but
  0.57 s slower than A. Report as "tested, not adopted".
- **Thinking on flash-lite (C vs D):** high is +2.7 points (p = 0.041, significant) but 1.04 s
  slower.
- Round 2 ran at the same time as the embedding end-to-end run (both call Gemini). The run order
  is shuffled, so any extra latency affects every setup equally.

## Router prompt fix and re-run (2026-11-07, freeze exception approved by the user)

Changes in `ORCHESTRATOR_PROMPT_HEAD` / examples: greetings, thanks and goodbyes go to general
with high confidence (example "หวัดดีครับ"); "paid X + how much tax / how much can I deduct" goes
to tax (example "ปีนี้จ่ายเบี้ยประกันสุขภาพไป 12,000 แล้วต้องเสียภาษีเท่าไร"). Neither example copies
an evaluation question.

Re-run: production setup only, 185 questions × 1 round (`results/router_thinking/prompt_v2/`).
- Accuracy 95.7% → **95.7%** (McNemar p = 1.0; 3 fixed, 3 now wrong).
- **Fixed:** "สวัสดีครับ" (ask-back menu at 0.3 → general 0.9), rpt_11, s35_route_expense_005.
- **Now wrong:** rpt_04 and rpt_02 (report → expense) and his_07 (→ planning). These
  report/expense boundary questions already flipped between rounds and setups with the old
  prompt, so this is noise, not a systematic change.
- **Unchanged as intended:** "จ่ายเบี้ยประกันสุขภาพไป 18,000 บาท" (paid only) → expense;
  "ขอบคุณมากนะ" → general; "แล้วอันนี้ลดหย่อนได้เท่าไหร่" → tax.
- Study A tax tasks on DeepSeek with the new prompt (`ollama_deepseek-v4.1-flash_prompt_v2/`,
  84 runs): tax_03 and tax_06 reach the tax agent in 18/18 runs (old prompt 9/16); tax-task
  success 77/84. The remaining failures are DeepSeek empty replies from the tax agent.
