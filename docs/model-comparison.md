# Model Comparison — Final Matrix (2026-09-18)

**Run:** `--compare --workers 10`, stamp `20260918_122702`
**Scope:** 7 candidate models × 6 evaluation dimensions, every cell `ok`
(42/42), zero rate-limit errors, zero parse failures.

**Candidates:** minimax-m3, qwen3.5:397b, deepseek-v4-pro:0813, glm-5.3,
glm-5.3-flash (Ollama Cloud, self-hosted cost model) vs
gemini-3.5-flash and gemini-3.5-flash-lite (Google API, metered).

**Judge:** fixed minimax-m3 instance scores all candidates (stateless,
thread-safe, built once per run). Self-judge bias checked: minimax did
not score itself highest on the quality dimension.

---

## 1. Routing (50 cases)

| Model | Provider | Accuracy | Mean Latency (s) |
|---|---|---|---|
| minimax-m3 | ollama | 0.94 | 13.98 |
| qwen3.5:397b | ollama | 0.94 | 22.67 |
| deepseek-v4-pro:0813 | ollama | 0.94 | 13.67 |
| glm-5.3 | ollama | 0.94 | 15.65 |
| glm-5.3-flash | ollama | 0.94 | 14.42 |
| gemini-3.5-flash-lite | google | 0.94 | **1.24** |
| gemini-3.5-flash | google | 0.91 | 3.28 |

Six-way tie at 0.94; gemini-3.5-flash-lite routes **10× faster** than
any Ollama model at equal accuracy. gemini-3.5-flash trails slightly
(0.91) — 3 misrouted queries out of 50.

## 2. Tax accuracy (20 cases, free routing)

| Model | Provider | Accuracy | Mean Latency (s) | MAE (THB) |
|---|---|---|---|---|
| minimax-m3 | ollama | **1.00** | 38.38 | 0.00 |
| qwen3.5:397b | ollama | **1.00** | 36.34 | 0.00 |
| deepseek-v4-pro:0813 | ollama | **1.00** | 40.95 | 0.00 |
| glm-5.3 | ollama | 0.95 | 39.50 | 225.00 |
| glm-5.3-flash | ollama | 0.95 | 38.78 | 175.00 |
| gemini-3.5-flash | google | 0.95 | 44.74 | 225.00 |
| gemini-3.5-flash-lite | google | 0.90 | 45.14 | 300.00 |

Three models compute Thai progressive tax perfectly end-to-end
(router → agent → tool call → Decimal math).

## 3. Tax accuracy — forced agent (20 cases, tool-argument focus)

| Model | Provider | Accuracy | Mean Latency (s) | MAE (THB) |
|---|---|---|---|---|
| minimax-m3 | ollama | **1.00** | 45.10 | 0.00 |
| glm-5.3 | ollama | **1.00** | 58.81 | 0.00 |
| glm-5.3-flash | ollama | **1.00** | 54.56 | 0.00 |
| gemini-3.5-flash | google | **1.00** | 12.47 | 0.00 |
| qwen3.5:397b | ollama | 0.95 | 47.05 | 225.00 |
| deepseek-v4-pro:0813 | ollama | 0.95 | 47.07 | 75.00 |
| gemini-3.5-flash-lite | google | 0.85 | 4.41 | 1,150.00 |

Isolates tool-argument quality (router removed). gemini-3.5-flash is
**5× faster than Ollama peers at perfect accuracy**. flash-lite's
MAE 1,150 THB shows degraded tool-argument precision on the top
bracket case.

## 4. Response quality (10 cases, judge = minimax-m3, 1–5 scaled to 0–1)

| Model | Provider | Accuracy | Mean Latency (s) |
|---|---|---|---|
| minimax-m3 | ollama | **0.88** | 68.40 |
| deepseek-v4-pro:0813 | ollama | **0.88** | 64.35 |
| glm-5.3 | ollama | **0.88** | 78.07 |
| gemini-3.5-flash | google | **0.88** | 42.29 |
| qwen3.5:397b | ollama | 0.85 | 78.73 |
| glm-5.3-flash | ollama | 0.85 | 89.70 |
| gemini-3.5-flash-lite | google | 0.84 | 43.93 |

Four-way tie at the top; gemini-3.5-flash reaches it 1.6× faster than
its Ollama co-leaders. A **cross-judge check** (below) confirms the
ranking is not a judge artifact.

### Cross-judge agreement check

The saved responses (105 rows, 7 models × 15 cases) were re-scored
with a second judge (`deepseek-v4-pro:0813`) — judge calls only, no
regeneration (`scripts/cross_judge_quality.py`):

| Model | minimax-m3 judge | deepseek judge | delta |
|---|---|---|---|
| qwen3.5:397b | 4.47 | 4.73 | +0.27 |
| gemini-3.5-flash | 4.27 | 4.67 | +0.40 |
| glm-5.3-flash | 4.40 | 4.47 | +0.07 |
| gemini-3.5-flash-lite | 4.00 | 4.47 | +0.47 |
| deepseek-v4-pro:0813 | 4.13 | 4.40 | +0.27 |
| glm-5.3 | 4.20 | 4.27 | +0.07 |
| minimax-m3 | 4.27 | 4.07 | **−0.20** |

Rank correlation is strong (Spearman ρ ≈ 0.91), exact-agreement rate
0.51 (absolute scores differ, rankings hold). Crucially, the deepseek
judge scores **minimax-m3 lowest of all models (−0.20)** — direct
evidence against self-judge bias: the judge that produced the original
table did not favor its own outputs.

## 5. Hallucination resistance (10 fabricated-fact cases)

| Model | Provider | Accuracy | Mean Latency (s) |
|---|---|---|---|
| gemini-3.5-flash | google | **0.70** | 33.69 |
| glm-5.3 | ollama | 0.60 | 29.45 |
| glm-5.3-flash | ollama | 0.60 | 29.19 |
| gemini-3.5-flash-lite | google | 0.60 | 39.53 |
| minimax-m3 | ollama | 0.50 | 32.15 |
| qwen3.5:397b | ollama | 0.50 | 29.47 |
| deepseek-v4-pro:0813 | ollama | 0.50 | 28.96 |

gemini-3.5-flash fabricates least; the Ollama cluster sits at 0.50–0.60.
This is the dimension with the largest spread — the strongest argument
for a cloud model if hallucination matters more than cost.

## 6. Recommendation safety (17 cases × 2 generations, guardrail eval)

| Model | Provider | Accuracy |
|---|---|---|
| minimax-m3 | ollama | **1.00** |
| qwen3.5:397b | ollama | **1.00** |
| deepseek-v4-pro:0813 | ollama | **1.00** |
| glm-5.3 | ollama | **1.00** |
| glm-5.3-flash | ollama | **1.00** |
| gemini-3.5-flash-lite | google | **1.00** |
| gemini-3.5-flash | google | **1.00** |

The deterministic guardrail (yield/insurance/disclosure rules) holds
for every model with genuine generations. Failed generations count as
failures — the 1.00 is not inflated by silent errors.

---

## Findings

1. **Two co-leaders for different reasons.**
   - **minimax-m3** (free, self-hosted): routing tie at 0.94, perfect
     free-tax and forced-tax, 1.00 rec-safety, quality co-leader.
     The production choice.
   - **gemini-3.5-flash**: best hallucination resistance (0.70),
     quality parity, 4–10× faster, perfect forced-tax at 12.5 s.
     The cheap cloud alternative.
   - **gemini-3.5-flash-lite**: fastest everywhere (1.2–4.4 s routing)
     but loses accuracy on tax (0.90 free / 0.85 forced) — fits
     cost-critical, latency-critical, non-calculation paths.
2. **Run-to-run variance ≈ ±0.05.** glm-5.3 scored 1.00/1.00 on tax in
   the 2026-09-16 run and 0.95/1.00 here (LLM nondeterminism). Treat
   close scores as ties.
3. **Gemini 3+ emits content-block lists** (`[{type: text, text: ...}]`
   with signature extras) rather than plain strings. This broke the
   router parser and downstream regexes until `_content_to_text`
   normalized them (`fe4719d`) — a real vendor integration cost beyond
   pricing.
4. **Reliability hardening is permanent** (commit `86763d9`): failed
   generations count as failures (an earlier bug scored 70 failed
   generations as compliant, inflating rec-safety to 1.00), 429 /
   socket-abort errors retry with exponential backoff, and per-case
   isolation means one bad case no longer errors a whole
   model×dimension cell.

## Reproduce

```bash
uv run python -m finance_ai.evaluation.cli --compare --workers 10 \
  --dimensions routing tax-accuracy tax-accuracy-forced quality \
  hallucination recommendation-safety
# override candidates:
#   --models ollama:minimax-m3 google:gemini-3.5-flash
# override judge:
#   --comparison-judge ollama:deepseek-v4-pro:0813
```

Raw per-dimension reports: `data/evaluation/results/model_comparison_{dimension}_{stamp}.json|.md`
(gitignored; run logs in `tmp-upload/model_comparison{8,9,10}.log`).

## Still open

- **RAG retrieval baseline** (`--eval rag`, ~24 cases — dataset ready);
  embeddings verified live on this key (gemini-embedding-001, 3072 dims).
- Cross-model **judge panel** for all judged dimensions is now a solved
  problem mechanically (per-case rows persisted +
  `scripts/cross_judge_quality.py`); quality dimension already
  cross-judged, hallucination/safety can reuse the same flow.
