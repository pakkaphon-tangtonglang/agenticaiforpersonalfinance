# Embedding study (4.11 C): design notes for the thesis (2026-11-07)

Use this when writing ch.4 (embedding comparison) and the portfolio report. The study has not
run yet; this file records **why each model was chosen** so the design can be explained.

## What is compared

Embedding models: gemini-embedding-001 (production), multilingual-e5-large-instruct, bge-m3,
paraphrase-multilingual-MiniLM-L12-v2 (the current fallback), plus BM25 keyword search as a
no-embedding baseline. All of them use the same knowledge base, the same chunks and one Chroma
index per model.

## Which parts use a chat model, and which model

| Part | Uses a chat model? | Model | Why |
|---|---|---|---|
| Retrieval scores (nDCG@10, Recall@5, MRR, query latency, index build time) | **No** | Embedding models only | This is the main result. It depends only on the embedding model, so the choice of chat model cannot change it. |
| End-to-end answers (top 3 embeddings) | Yes, to write the answer | **gemini-3.5-flash** (production) | The results then describe the real system. A stronger model is not better here: it can answer from its own knowledge when retrieval returns the wrong chunks, which hides the retrieval differences the study is meant to measure. |
| Judging the answers (faithful to the cited source, correct) | Yes, as the judge | **deepseek-v4.1-flash** (Ollama Cloud, paid credit) | A different model family from the one writing the answers, because models tend to rate their own answers higher (self-preference bias). Changed from gpt-oss:120b on 2026-11-07 because gpt-oss is weak in Thai; DeepSeek's Thai was natural in the Study A pilot. The judge is checked against hand labels, and later against the finance experts' labels, before its scores are used. |

Cost (estimate): about $2 in total. 360 Gemini answers make up most of it; the 360 DeepSeek judgements cost about $0.15.

## Suggested thesis text (Thai, ch.4)

> การทดลองเปรียบเทียบโมเดลฝังข้อความ (Embedding Model) แบ่งเป็น 2 ส่วน ส่วนแรกวัดคุณภาพการค้นคืนเอกสาร
> ด้วย nDCG@10, Recall@5 และ MRR ซึ่งขึ้นกับโมเดลฝังข้อความเพียงอย่างเดียว ไม่มีโมเดลภาษาขนาดใหญ่เข้ามาเกี่ยวข้อง
> การเลือกโมเดลภาษาจึงไม่มีผลต่อผลลัพธ์หลักของการทดลอง ส่วนที่สองวัดคำตอบแบบครบวงจร (End-to-End) ของ 3 โมเดล
> ที่ค้นคืนได้ดีที่สุด โดยใช้ gemini-3.5-flash ซึ่งเป็นโมเดลที่ระบบใช้งานจริงเป็นผู้ตอบ เพื่อให้ผลสะท้อนพฤติกรรมของระบบจริง
> คณะผู้จัดทำไม่เลือกโมเดลที่ใหญ่กว่า เพราะโมเดลที่เก่งกว่าอาจตอบจากความรู้ของตัวเองได้แม้ระบบค้นคืนเอกสารผิด
> ซึ่งจะบดบังความแตกต่างระหว่างโมเดลฝังข้อความที่ต้องการวัด ส่วนการให้คะแนนคำตอบใช้ deepseek-v4.1-flash เป็นผู้ตัดสิน
> (LLM-as-a-Judge) ซึ่งเป็นโมเดลคนละตระกูลกับผู้ตอบ เพื่อลดอคติที่โมเดลมักให้คะแนนคำตอบของตัวเองสูงกว่าความเป็นจริง
> และคณะผู้จัดทำตรวจความสอดคล้องของผู้ตัดสินกับการประเมินของผู้เชี่ยวชาญด้านการเงินก่อนนำคะแนนไปใช้

## English (portfolio report)

> Retrieval quality (nDCG@10, Recall@5, MRR) depends only on the embedding model; no chat model is
> involved, so the main result cannot be changed by the choice of LLM. End-to-end answers for the
> top 3 embeddings are written by gemini-3.5-flash, the production model, so the numbers describe
> the real system. A stronger answer model would be worse for this purpose: it can answer from its
> own knowledge when retrieval fails, masking the retrieval differences under test. Answers are
> judged by deepseek-v4.1-flash, a different model family, to avoid self-preference bias, and the judge
> is validated against finance-expert labels before its scores are used.

## Status (2026-11-07)

- Dataset done: `data/evaluation/rag_v2_dataset.yaml` (120 questions; every question has at least one relevant chunk; leakage median 0.18, 5 questions above 0.5).
- Code: `src/finance_ai/evaluation/embedding_study/` (study.py = retrieval, end_to_end.py = answers + judge; 25 tests).

## Still to do before the run (original list)

- Build a 120-question RAG dataset (the current set has 39). Check that no question copies
  wording from the source documents (leakage).
- Install `sentence-transformers` for the local models (e5, bge-m3, MiniLM).
- Ask the user before any spend (Gemini embeddings and answers: a few THB).

## Judge check (2026-11-07, first 10 questions × 3 retrievers = 30 answers)

I read each answer against its 3 passages and the gold passage, then compared my verdicts with
DeepSeek's. This is a sanity check by another model (Claude), not human validation; the finance
experts' labels remain the real check.

- **Correct:** agree on 30/30 (all answers correct).
- **Faithful:** agree on 28/30. The judge marked q002 unfaithful twice (gemini, bge_m3), but the
  PIT90 passage does state the parents' 30,000 income limit, so these are judge false negatives.
  The other 4 "unfaithful" verdicts (q001 gemini, q003 bge_m3, q007 e5, q009 e5) are strict but
  defensible: each answer adds a detail not in the passages (filing months, institution types,
  form names, the "only one child may claim" rule).
- **Reading:** the judge is strict on faithfulness, and equally strict for every retriever, so
  the comparison between retrievers is fair. Absolute faithfulness rates are a lower bound.

## Retrieval results (2026-11-07)

`data/evaluation/results/embedding_study/report.md`: gemini nDCG@10 0.805 / Recall@3 0.900;
bge-m3 0.724 / 0.842; e5-large-instruct 0.722 / 0.825; BM25 0.568 / 0.683; MiniLM 0.394 / 0.433.
Gemini is significantly better than every other model (sign test p ≤ 0.0002).
**MiniLM, the configured fallback (`rag_sentence_transformer_model`), is the worst, below
keyword search.** bge-m3 is the better local fallback (0.724, 50 ms per query on CPU). Changing
the fallback is a config change (freeze exception), so list it as a recommendation or future work.
e5 loaded in float16 on CPU (slow index build, 1,238 s). This may cost it a little precision.

## End-to-end results (2026-11-07, 120 questions × 3 retrievers, 0 errors)

`data/evaluation/results/embedding_study/end_to_end.md` (raw rows: `end_to_end.jsonl`).

| Retriever | Correct (95% CI) | Faithful (95% CI) |
|---|---|---|
| gemini | 93.3% (87.4–96.6) | 88.3% (81.4–92.9) |
| bge-m3 | 94.2% (88.4–97.1) | 82.5% (74.7–88.3) |
| e5-large-instruct | 92.5% (86.4–96.0) | 86.7% (79.4–91.6) |

- **Correctness:** no significant differences (McNemar p = 1.0 for both against gemini). The
  retrieval gap (nDCG 0.805 vs about 0.72) mostly disappears in the final answer, because the
  top 3 chunks usually still contain an answering passage and the answer model copes.
- **Faithfulness:** gemini is highest, but neither difference is significant (bge-m3 7 vs 14
  discordant, p = 0.19; e5 10 vs 12, p = 0.83).
- **Thesis reading:** gemini-embedding-001 retrieves significantly better. In end-to-end answers,
  a good local model (bge-m3) is about as correct, so bge-m3 is a sound fallback. MiniLM is not.
- **Failures worth naming:** 3 empty Gemini answers (q012, q020, q082). q036 failed for all three
  retrievers: the answers did not give the guide's "if unsure, use Snowball" recommendation.
  q040 failed for all three: the answers mixed the age-band saving rates.
- **Judge:** the 30-answer check above applies (28/30 agreement on faithfulness). The judge's
  incorrect verdicts were not all re-read; the expert review is the remaining validation.
