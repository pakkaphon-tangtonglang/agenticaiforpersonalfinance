# Known Issues & Follow-ups

Notes for future sessions. Last updated: router fix `2ecbb15` (2026 session).

## 1. Router: add Thai few-shot examples if mis-routing persists

**Status:** deferred (user chose not to include in the `2ecbb15` fix).

The router improvements shipped in `2ecbb15` (chat-history context,
confidence clarify-back, pre-route symbol search, general-chat dispatch fix)
cover most confusion cases. If specific Thai phrasings still mis-route:

**Next lever:** add 3–4 few-shot examples per category to
`ORCHESTRATOR_SYSTEM_PROMPT` in `src/finance_ai/agents/prompts.py` (~line 53).

Format that works well for minimax-m3:

```
ตัวอย่าง:
- "จ่ายค่ากาแฟ 80 บาท" → {"intent": "expense", "confidence": 0.95}
- "อยากออมเงิน 100,000 บาท" → {"intent": "planning", "confidence": 0.9}
- "อันนั้นล่ะ" (หลังจากคุยเรื่องหุ้น) → {"intent": "asset_monitoring", "confidence": 0.85}
```

Known overlap zones to cover with examples: expense vs planning (savings
statements), report vs recommendation vs general.

Also useful: `evaluation/architecture_benchmark.py` already prototypes a
two-stage macro→specific routing experiment — run it to measure before/after.

## 2. Price data empty on Render for SET stocks

**Status:** open (not a routing issue — observed during live verification
of `2ecbb15`).

Live test `"ราคา PTT เท่าไหร่"` on Render routed correctly to
`asset_monitoring` and resolved `PTT.BK`, but the price tool returned
empty data ("ข้อมูลทุกช่องว่างเปล่า"). The same query worked locally earlier
in development, so suspect:

- Yahoo Finance rate-limiting/blocking Render's outbound IPs (free tier,
  shared egress), or
- missing User-Agent / headers on the server-side request
  (see `src/finance_ai/tools/price_client.py`, `market_data_service.py`,
  `symbol_search_service.py` — the search service already sets a
  User-Agent; check the price client does too), or
- yfinance cache/cookie expiry in the Render instance.

Reproduce: `curl -X POST https://personal-finance-ai-wcr0.onrender.com/chat -H "Content-Type: application/json" -d '{"query":"ราคา PTT เท่าไหร่","user_id":"test"}'`

Check price fetch in isolation first (logs / a direct call to
`market_data_service`) before touching the agent graph.
