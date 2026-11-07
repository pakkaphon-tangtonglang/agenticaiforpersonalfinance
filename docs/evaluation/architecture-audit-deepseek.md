# Architecture study: manual audit, deepseek-v4.1-flash (3 rounds, 2026-11-07)

I read every run the automatic checks failed (73 of 720). I also scanned passing runs of the "last month" tasks for a wrong month name: there were 9 hits, all fine (the reports cover October; only their creation date says November).
Results: `data/evaluation/results/architecture_v2/ollama_deepseek-v4.1-flash/` (`report.md`, `audit.yaml`). Same tasks, scorer and judgement rules as the gpt-oss audit (`architecture-audit.md`).

## Overridden to pass (6)

- rpt_04 / single_agent / round 1: Correct net 37,070 for October, from the expense and income summary tools instead of the report tool (same data; equivalent tools were accepted for rpt_05 in the gpt-oss audit).
- rpt_06 / single_agent / round 1: Full Q4 summary (income, spending by category, portfolio, health score) from cross-agent summary tools instead of the report tool.
- rpt_04 / single_agent / round 2: Correct net 37,070 for October from the expense and income summary tools (same as round 1).
- rpt_04 / single_agent / round 3: Correct net 37,070 for October from the cross-agent expense and income summaries (same as round 1).
- crs_02 / single_agent / round 2: October spending (12,930) plus the user's recorded deductions from calculate_tax_from_records: grounded in the user's data, not model memory.
- crs_02 / single_agent / round 3: October spending (12,930) plus a table of recorded deductions, caps and remaining room from calculate_tax_from_records.

## Real failures (67 of 720)

| Task | Design | Round | Failure type | Route | What happened |
|---|---|---:|---|---|---|
| crs_02 | hierarchical | 1 | empty_answer | tax | Empty reply from the model. |
| crs_02 | hierarchical | 2 | wrong_route | clarify | Level 1 was unsure and asked back on a clear question. |
| crs_02 | hierarchical | 3 | empty_answer | tax | Empty reply from the model. |
| crs_02 | peer_to_peer | 2 | wrong_route | report | Ended at the report agent (spending right, deductions from the knowledge base): the final agent is report, not tax or expense. |
| crs_02 | peer_to_peer | 3 | wrong_route | report → clarify | Report agent asked back on a clear question. |
| crs_03 | hierarchical | 3 | missing_tool | recommendation | Routed to recommendation; health score and recommendations, without the portfolio tool. |
| crs_03 | hub_spoke | 1 | wrong_route | clarify | Router was unsure and asked back on a clear portfolio question. |
| crs_03 | hub_spoke | 2 | missing_tool | recommendation | Routed to recommendation; health score and recommendations, without the portfolio tool. |
| crs_05 | hierarchical | 1 | missing_tool | tax | Tax correct (36,500) but no saving-plan calculation: the tax agent has no planning tool. |
| crs_05 | hierarchical | 2 | missing_tool | tax | Tax correct (36,500) but no saving-plan calculation: the tax agent has no planning tool. |
| crs_05 | hierarchical | 3 | missing_tool | tax | Tax correct (36,500) but no saving-plan calculation: the tax agent has no planning tool. |
| crs_05 | hub_spoke | 1 | missing_tool | tax | Tax correct (36,500) but no saving-plan calculation: the tax agent has no planning tool. |
| crs_05 | hub_spoke | 2 | missing_tool | tax | Tax correct (36,500) but no saving-plan calculation: the tax agent has no planning tool. |
| crs_05 | hub_spoke | 3 | missing_tool | tax | Tax correct (36,500) but no saving-plan calculation: the tax agent has no planning tool. |
| crs_05 | peer_to_peer | 1 | missing_tool | asset_monitoring → tax | Forwarded to tax; Tax correct (36,500) but no saving-plan calculation: the tax agent has no planning tool. |
| crs_05 | peer_to_peer | 2 | missing_tool | asset_monitoring → tax | Forwarded to tax; Tax correct (36,500) but no saving-plan calculation: the tax agent has no planning tool. |
| crs_05 | peer_to_peer | 3 | missing_tool | asset_monitoring → tax | Forwarded to tax; Tax correct (36,500) but no saving-plan calculation: the tax agent has no planning tool. |
| crs_07 | hierarchical | 1 | missing_tool | recommendation | Health score, recommendations and the 3-6 month rule from the knowledge base, then asked the user for the current emergency fund (already stored as a goal). |
| crs_07 | hierarchical | 2 | missing_tool | recommendation | Health score, recommendations and the 3-6 month rule from the knowledge base, then asked the user for the current emergency fund (already stored as a goal). |
| crs_07 | hierarchical | 3 | missing_tool | recommendation | Health score, recommendations and the 3-6 month rule from the knowledge base, then asked the user for the current emergency fund (already stored as a goal). |
| crs_07 | hub_spoke | 1 | missing_tool | recommendation | Health score, recommendations and the 3-6 month rule from the knowledge base, then asked the user for the current emergency fund (already stored as a goal). |
| crs_07 | hub_spoke | 2 | missing_tool | recommendation | Health score, recommendations and the 3-6 month rule from the knowledge base, then asked the user for the current emergency fund (already stored as a goal). |
| crs_07 | hub_spoke | 3 | missing_tool | recommendation | Health score, recommendations and the 3-6 month rule from the knowledge base, then asked the user for the current emergency fund (already stored as a goal). |
| crs_07 | peer_to_peer | 1 | missing_tool | recommendation | Health score, recommendations and the 3-6 month rule from the knowledge base, then asked the user for the current emergency fund (already stored as a goal). |
| crs_07 | peer_to_peer | 2 | missing_tool | recommendation | Health score, recommendations and the 3-6 month rule from the knowledge base, then asked the user for the current emergency fund (already stored as a goal). |
| crs_07 | peer_to_peer | 3 | missing_tool | recommendation | Health score, recommendations and the 3-6 month rule from the knowledge base, then asked the user for the current emergency fund (already stored as a goal). |
| crs_08 | hierarchical | 1 | missing_tool | recommendation | Routed to recommendation; built the 'monthly report' from the health score and recommendations, without the report tool. |
| crs_08 | hierarchical | 3 | missing_tool | recommendation | Routed to recommendation; built the 'monthly report' from the health score and recommendations, without the report tool. |
| crs_08 | hub_spoke | 1 | missing_tool | recommendation | Routed to recommendation; built the 'monthly report' from the health score and recommendations, without the report tool. |
| crs_09 | hierarchical | 1 | missing_tool | tax | Tax correct (11,500) but no RMF holding value: the tax agent has no portfolio tool. |
| crs_09 | hierarchical | 2 | missing_tool | tax | Tax correct (11,500) but no RMF holding value: the tax agent has no portfolio tool. |
| crs_09 | hierarchical | 3 | missing_tool | tax | Tax correct (11,500) but no RMF holding value: the tax agent has no portfolio tool. |
| crs_09 | hub_spoke | 1 | missing_tool | tax | Tax correct (11,500) but no RMF holding value: the tax agent has no portfolio tool. |
| crs_09 | hub_spoke | 2 | missing_tool | tax | Tax correct (11,500) but no RMF holding value: the tax agent has no portfolio tool. |
| crs_09 | hub_spoke | 3 | missing_tool | tax | Tax correct (11,500) but no RMF holding value: the tax agent has no portfolio tool. |
| crs_09 | peer_to_peer | 1 | missing_tool | tax | Tax correct (11,500) but no RMF holding value: the tax agent has no portfolio tool. |
| crs_09 | peer_to_peer | 2 | missing_tool | tax | Tax correct (11,500) but no RMF holding value: the tax agent has no portfolio tool. |
| crs_09 | peer_to_peer | 3 | missing_tool | tax | Tax correct (11,500) but no RMF holding value: the tax agent has no portfolio tool. |
| crs_12 | hierarchical | 2 | wrong_route | clarify | Router or entry agent was unsure and asked back on a clear question. |
| crs_12 | hub_spoke | 2 | empty_answer | tax | Empty reply from the model. |
| crs_12 | hub_spoke | 3 | wrong_route | clarify | Router or entry agent was unsure and asked back on a clear question. |
| crs_12 | peer_to_peer | 2 | wrong_route | planning → clarify | Planning agent asked back on a clear question. |
| pln_07 | hierarchical | 1 | wrong_route | general | Routed to general chat; the answer (3-6 months) was right but came from the model, not the knowledge base. |
| pln_07 | hierarchical | 2 | wrong_route | general | Routed to general chat; the answer (3-6 months) was right but came from the model, not the knowledge base. |
| pln_07 | hierarchical | 3 | wrong_route | general | Routed to general chat; the answer (3-6 months) was right but came from the model, not the knowledge base. |
| pln_07 | hub_spoke | 1 | wrong_route | general | Routed to general chat; the answer (3-6 months) was right but came from the model, not the knowledge base. |
| pln_07 | hub_spoke | 2 | wrong_route | general | Routed to general chat; the answer (3-6 months) was right but came from the model, not the knowledge base. |
| pln_07 | hub_spoke | 3 | wrong_route | general | Routed to general chat; the answer (3-6 months) was right but came from the model, not the knowledge base. |
| rec_03 | peer_to_peer | 1 | wrong_route | tax → asset_monitoring → recommendation → asset_monitoring | Bounced tax, asset, recommendation, asset; the final asset agent gave portfolio-based advice (wrong final agent). |
| rec_03 | single_agent | 1 | missing_tool | — | Asked for goal, horizon and risk level instead of using the user's data. |
| rec_06 | hierarchical | 1 | wrong_route | asset_monitoring | Routed to asset monitoring; good portfolio-based answer but no psychological-cue check (wrong agent). |
| rec_06 | hierarchical | 2 | wrong_route | asset_monitoring | Routed to asset monitoring; good portfolio-based answer but no psychological-cue check (wrong agent). |
| rec_06 | hierarchical | 3 | wrong_route | asset_monitoring | Routed to asset monitoring; good portfolio-based answer but no psychological-cue check (wrong agent). |
| rec_06 | peer_to_peer | 2 | wrong_route | planning → asset_monitoring → recommendation → asset_monitoring | Bounced planning, asset, recommendation, asset; the final asset agent gave a portfolio-based answer (wrong final agent). |
| rec_06 | peer_to_peer | 3 | wrong_route | planning → asset_monitoring → recommendation → asset_monitoring | Bounced planning, asset, recommendation, asset; the final asset agent gave a portfolio-based answer (wrong final agent). |
| tax_03 | hierarchical | 2 | wrong_route | expense | Routed to expense (the prompt's paid-premium rule); recorded the premium and estimated tax by hand. |
| tax_03 | hub_spoke | 1 | wrong_route | expense | Router sent 'paid life insurance 50,000, how much tax' to expense (the prompt's paid-premium rule); tax estimated by hand without the calculator. |
| tax_03 | hub_spoke | 2 | wrong_route | expense | Router sent 'paid life insurance 50,000, how much tax' to expense (the prompt's paid-premium rule); tax estimated by hand without the calculator. |
| tax_03 | hub_spoke | 3 | wrong_route | expense | Router sent 'paid life insurance 50,000, how much tax' to expense (the prompt's paid-premium rule); tax estimated by hand without the calculator. |
| tax_06 | hierarchical | 1 | wrong_route | expense | Routed to expense (paid health insurance); computed tax by hand and forgot the 100,000 expense deduction: wrong value. |
| tax_06 | hierarchical | 2 | wrong_route | expense | Routed to expense (paid health insurance); computed tax by hand and forgot the 100,000 expense deduction: wrong value. |
| tax_06 | hierarchical | 3 | wrong_route | expense | Routed to expense (paid health insurance); computed tax by hand and forgot the 100,000 expense deduction: wrong value. |
| tax_06 | hub_spoke | 2 | wrong_route | expense | Routed to expense (paid-premium rule); recorded the deduction and estimated tax without the calculator. |
| tax_07 | hierarchical | 3 | empty_answer | tax | Empty reply from the model. |
| tax_07 | hub_spoke | 1 | empty_answer | tax | Empty reply from the model. |
| tax_07 | hub_spoke | 2 | empty_answer | tax | Empty reply from the model. |
| vag_04 | single_agent | 1 | did_not_ask_back | — | Asked back, but in Chinese. |

## Patterns compared with gpt-oss:20b

- **The same structural failures remain:** crs_05, crs_07 and crs_09 fail in every routed design. The routed agent lacks the other domain's tool, so a stronger model does not fix them. The single agent, which has every tool, solved all 36 cross-topic runs (12 tasks x 3 rounds) and is significantly better than hub-and-spoke overall (98.9% vs 87.2%, McNemar p < 0.0001).
- **Peer-to-peer is no longer the weakest design.** DeepSeek's entry agents forward correctly (96.7% route accuracy over 3 rounds, vs 85.6% with gpt-oss). P2P's routing weakness was mostly a weak-model effect.
- **A new router failure:** tax questions that mention a paid premium went to expense, because of the paid-premium rule added in commit f94d1f7 (after the gpt-oss run). The router prompt was fixed on 2026-11-07, and a re-run of the 7 tax tasks (`ollama_deepseek-v4.1-flash_prompt_v2`) sent tax_03 and tax_06 to tax in 18/18 runs.
- **Model quirks:** empty replies (6 runs, all from the tax agent: it produced about 500 tokens but no visible text) and 1 ask-back in Chinese.
