# Architecture study: manual audit (gpt-oss:20b, 3 rounds, 2026-11-02 to 2026-11-03)

I read every failed run by hand after automatic scoring. Every row below is a real failure.
Results: `data/evaluation/results/architecture_v2/ollama_gpt-oss_20b/`
(`report.md`, `audit.yaml`). Rounds are numbered 1-3.

## How the audit was done

1. Automatic scoring: route, required tool groups, expected values, database rows.
2. I read every failed run and recorded the reason (table below).
3. Passing runs were checked for one known blind spot. Every seeded month has the
   same amounts, so a reply that names the wrong month still passes the value
   checks. I searched the passing runs of "last month" tasks for other month names
   and read every hit. Six runs fetched October correctly but told the user
   "September" or "November". They are overridden to fail in `audit.yaml`.
   Other kinds of false pass were not searched for, so they may remain.

## Corrections made after round 1 (applied to all four designs and all rounds)

- **Scorer:** numbers with a space or narrow space as the thousands separator
  (`60 000`) now match, and so do non-breaking hyphens in fund codes.
- **Task definitions:** rec_01, rec_02, rec_04 and pln_03 also accept the report agent,
  because its report contains the same health score, recommendations and goal table.
  rpt_05 also accepts the expense agent's category summary. In crs_08 the report tool
  also covers the recommendations part, because the report has a recommendations section.

## Judgement rules used

- **Grounding counts.** A knowledge answer given from model memory without the
  knowledge base fails, even when it is correct (crs_12 / single_agent / round 3).
  Grounded answers are a thesis claim.
- **Asking for data the system already has fails.** Examples are income, goals and
  the emergency fund.
- **A wrong month shown to the user fails,** even when the numbers are right.
- **Asking for risk level on "what should I invest in" (rec_03) is counted as a
  failure,** because the recommendation tools can use the user's stored data.
  A human advisor might ask first, so this is the most debatable rule.

## Failed runs (131 of 720)

| Task | Design | Round | Failure type | Route | What happened |
|---|---|---:|---|---|---|
| ast_01 | hierarchical | 1 | wrong_route | clarify | Level 1 was unsure and asked back on a clear stock-price question. |
| ast_04 | hierarchical | 2 | wrong_route | clarify | Level 1 was unsure and asked back on a clear stock-news question. |
| ast_04 | hierarchical | 3 | wrong_route | clarify | Level 1 was unsure and asked back on a clear stock-news question. |
| ast_05 | hierarchical | 2 | wrong_route | recommendation | Routed to recommendation; listed SET50 funds from the knowledge base instead of the live fund search. |
| ast_05 | hub_spoke | 1 | wrong_route | recommendation | Routed to recommendation; listed SET50 funds from the knowledge base instead of the live fund search. |
| ast_05 | hub_spoke | 3 | wrong_route | recommendation | Routed to recommendation; listed SET50 funds from the knowledge base instead of the live fund search. |
| ast_05 | peer_to_peer | 2 | wrong_route | tax → recommendation | Tax forwarded to recommendation, which listed SET50 funds from the knowledge base instead of the live fund search. |
| ast_05 | peer_to_peer | 3 | wrong_route | tax → recommendation | Tax forwarded to recommendation, which listed SET50 funds from the knowledge base instead of the live fund search. |
| ast_06 | hierarchical | 1 | wrong_route | expense | Routed to expense; saved the share purchase as a 60,000 THB expense instead of a portfolio holding. |
| ast_06 | hierarchical | 2 | wrong_route | clarify | Level 1 was unsure and asked back on a clear buy-and-record request. |
| ast_06 | single_agent | 1 | missing_tool | — | Resolved the symbol, then asked for confirmation instead of saving the holding. |
| crs_01 | hub_spoke | 1 | missing_tool | planning | Planning read spending and income but not the goal, then asked the user for the goal amount the system already has. |
| crs_01 | hub_spoke | 2 | missing_tool | planning | Planning read spending and income but not the goal; asked the user for the goal amount the system already has. |
| crs_01 | hub_spoke | 3 | missing_tool | planning | Planning read spending and income but not the goal; asked the user for the goal amount the system already has. |
| crs_01 | peer_to_peer | 1 | wrong_route | recommendation → clarify | Entry agent asked back on a clear question. |
| crs_01 | peer_to_peer | 2 | wrong_route | recommendation → planning → clarify | Recommendation forwarded to planning, then asked back on a clear question. |
| crs_01 | single_agent | 1 | missing_tool | — | Asked the user for goal details instead of reading them. |
| crs_01 | single_agent | 2 | missing_tool | — | Asked the user for income and goal details instead of reading them. |
| crs_02 | hierarchical | 3 | empty_answer | tax | Empty reply from the model. |
| crs_02 | hub_spoke | 1 | missing_tool | tax | Tax agent gave this month's spending only; no deduction part. |
| crs_02 | hub_spoke | 2 | missing_tool | tax | Spending correct (October, 12,930); deduction list from model memory without a knowledge-base lookup, and it mixes up SSF with the provident fund. |
| crs_02 | hub_spoke | 3 | missing_tool | tax | Reported this month's partial spending (9,060, labelled November) instead of October; deduction part from model memory. |
| crs_02 | peer_to_peer | 1 | wrong_route | report → clarify | Entry agent asked back on a clear question. |
| crs_02 | peer_to_peer | 3 | wrong_route | report → expense → tax → clarify | Report, expense, tax, then asked back on a clear question. |
| crs_02 | single_agent | 1 | missing_tool | — | Spending summary only; no deduction part. |
| crs_02 | single_agent | 2 | missing_tool | — | Spending table labelled as the current month; deduction list from model memory without a lookup. |
| crs_02 | single_agent | 3 | missing_tool | — | Spending table labelled November; deduction list from model memory without a lookup. |
| crs_03 | peer_to_peer | 3 | missing_tool | tax → recommendation → asset_monitoring | Showed the portfolio and its own rebalancing notes, without the recommendation tool or the knowledge base. |
| crs_03 | single_agent | 1 | missing_tool | — | Showed the portfolio but gave no rebalancing advice. |
| crs_03 | single_agent | 3 | missing_tool | — | Showed the portfolio and its own rebalancing notes, without the recommendation tool or the knowledge base. |
| crs_05 | hierarchical | 1 | missing_tool | tax | Tax computed but no saving plan: the tax agent has no planning tool. |
| crs_05 | hierarchical | 2 | missing_tool | tax | Tax computed but no saving plan: the tax agent has no planning tool. |
| crs_05 | hierarchical | 3 | missing_tool | tax | Tax computed but no saving plan: the tax agent has no planning tool. |
| crs_05 | hub_spoke | 1 | missing_tool | tax | Tax computed but no saving plan: the tax agent has no planning tool. |
| crs_05 | hub_spoke | 2 | missing_tool | tax | Tax computed but no saving plan: the tax agent has no planning tool. |
| crs_05 | hub_spoke | 3 | missing_tool | tax | Tax computed but no saving plan: the tax agent has no planning tool. |
| crs_05 | peer_to_peer | 1 | missing_tool | asset_monitoring → tax | Forwarded to tax; Tax computed but no saving plan: the tax agent has no planning tool. |
| crs_05 | peer_to_peer | 2 | missing_tool | asset_monitoring → tax | Forwarded to tax; Tax computed but no saving plan: the tax agent has no planning tool. |
| crs_05 | peer_to_peer | 3 | missing_tool | asset_monitoring → tax | Forwarded to tax; Tax computed but no saving plan: the tax agent has no planning tool. |
| crs_05 | single_agent | 2 | missing_tool | — | Tax correct (36,500) but no saving-plan calculation. |
| crs_06 | single_agent | 1 | missing_tool | — | Quoted 58.00 THB (the stored holding price) instead of the current 33.25; never called the price tool. |
| crs_07 | hierarchical | 1 | missing_tool | recommendation | Health score only; did not work out the emergency-fund gap. |
| crs_07 | hierarchical | 2 | missing_tool | recommendation | Health score and a 6-month target, then asked the user for the current emergency fund (already stored as a goal). |
| crs_07 | hierarchical | 3 | missing_tool | recommendation | Health score and recommendations only; no emergency-fund gap. |
| crs_07 | hub_spoke | 1 | missing_tool | recommendation | Health score and recommendations; did not work out the emergency-fund gap. |
| crs_07 | hub_spoke | 2 | missing_tool | recommendation | Health score, then asked the user for spending and the current emergency fund the system already has. |
| crs_07 | hub_spoke | 3 | missing_tool | recommendation | Asked the user for income and savings instead of reading them. |
| crs_07 | peer_to_peer | 1 | wrong_route | recommendation → clarify | Entry agent asked back on a clear question. |
| crs_07 | peer_to_peer | 2 | wrong_route | recommendation → clarify | Entry agent asked back on a clear question. |
| crs_07 | peer_to_peer | 3 | wrong_route | recommendation → clarify | Entry agent asked back on a clear question. |
| crs_08 | hierarchical | 1 | missing_tool | recommendation | Routed to recommendation; gave recommendations but no monthly report. |
| crs_08 | hierarchical | 3 | missing_tool | recommendation | Routed to recommendation; built a 'report' from the recommendations with wrong figures (income 1,200 THB/month); no report tool. |
| crs_08 | peer_to_peer | 2 | wrong_route | report → clarify | Entry agent asked back on a clear question. |
| crs_08 | peer_to_peer | 3 | missing_tool | report → recommendation | Report forwarded to recommendation; recommendations only, no monthly report. |
| crs_09 | hierarchical | 1 | missing_tool | tax | Tax computed (11,500 correct) but RMF holding value missing: the tax agent has no portfolio tool. |
| crs_09 | hierarchical | 2 | missing_tool | tax | Tax computed (11,500 correct) but RMF holding value missing: the tax agent has no portfolio tool. |
| crs_09 | hierarchical | 3 | missing_tool | tax | Tax computed, then asked the user for the RMF value (already in the portfolio). |
| crs_09 | hub_spoke | 1 | missing_tool | tax | Tax computed (11,500 correct) but RMF holding value missing: the tax agent has no portfolio tool. |
| crs_09 | hub_spoke | 2 | missing_tool | tax | Tax computed (11,500 correct) but RMF holding value missing: the tax agent has no portfolio tool. |
| crs_09 | hub_spoke | 3 | missing_tool | tax | Tax computed (11,500 correct) but RMF holding value missing: the tax agent has no portfolio tool. |
| crs_09 | peer_to_peer | 1 | missing_tool | tax | Tax computed (11,500 correct) but RMF holding value missing: the tax agent has no portfolio tool. |
| crs_09 | peer_to_peer | 2 | empty_answer | tax | Empty reply from the model. |
| crs_09 | peer_to_peer | 3 | missing_tool | tax | Tax computed (11,500 correct) but RMF holding value missing: the tax agent has no portfolio tool. |
| crs_09 | single_agent | 1 | missing_tool | — | Misread the question as selling RMF; no holding value. |
| crs_09 | single_agent | 2 | missing_tool | — | Asked for the RMF fund name instead of reading the portfolio; no tax calculation. |
| crs_09 | single_agent | 3 | missing_tool | — | Asked for the RMF fund name instead of reading the portfolio; no tax calculation. |
| crs_10 | single_agent | 1 | wrong_write | — | Passed the 40,000 deposit as the new total: the emergency fund dropped from 96,000 to 40,000 (model misuse despite the added_amount fix). |
| crs_11 | hierarchical | 1 | wrong_route | clarify | Level 1 was unsure and asked back on a clear question. |
| crs_12 | hierarchical | 1 | missing_tool | tax | Answered the deduction part only; did not look up last month's health spending. |
| crs_12 | hierarchical | 2 | empty_answer | tax | Empty reply from the model. |
| crs_12 | hierarchical | 3 | empty_answer | tax | Empty reply from the model. |
| crs_12 | hub_spoke | 1 | empty_answer | tax | Empty reply from the model. |
| crs_12 | hub_spoke | 2 | missing_tool | tax | Deduction rule from the knowledge base, then asked the user for last month's health spending (recorded: 300). |
| crs_12 | hub_spoke | 3 | missing_tool | tax | Deduction rule from the knowledge base, then asked the user for last month's health spending (recorded: 300). |
| crs_12 | peer_to_peer | 1 | empty_answer | planning → tax | Empty reply from the model. |
| crs_12 | peer_to_peer | 2 | missing_tool | planning → tax | Near-empty reply ('!'). |
| crs_12 | peer_to_peer | 3 | missing_tool | planning → tax → expense → tax | Planning, tax, expense, tax; deduction rule only, asked the user for the spending. |
| crs_12 | single_agent | 1 | missing_tool | — | Answered the deduction part only. |
| crs_12 | single_agent | 3 | missing_tool | — | Both parts correct (300 THB), but the deduction rule came from model memory, not the knowledge base (grounding required). |
| exp_04 | peer_to_peer | 2 | wrong_route | recommendation → clarify | Entry agent asked back on a clear question. |
| exp_05 | hub_spoke | 1 | audit_fail | expense | Audit override: Fetched October correctly but told the user it was November. |
| exp_05 | hub_spoke | 2 | audit_fail | expense | Audit override: Fetched October correctly but told the user it was November. |
| exp_05 | peer_to_peer | 2 | audit_fail | report → expense | Audit override: Fetched October correctly but told the user it was November. |
| pln_01 | peer_to_peer | 3 | wrong_route | planning → clarify | Planning, the right agent, chose to ask back on a clear question. |
| pln_02 | peer_to_peer | 2 | wrong_route | recommendation → planning → clarify | Recommendation forwarded to planning, then asked back on a clear question. |
| pln_02 | peer_to_peer | 3 | wrong_route | recommendation → planning → clarify | Recommendation forwarded to planning, then asked back on a clear question. |
| pln_03 | peer_to_peer | 1 | wrong_route | report → clarify | Forwarded to report, which asked back on a clear question. |
| pln_03 | peer_to_peer | 3 | wrong_route | report → clarify | Entry agent asked back on a clear question. |
| pln_06 | peer_to_peer | 1 | wrong_route | asset_monitoring → recommendation → clarify | Bounced between agents, then asked back. |
| pln_06 | peer_to_peer | 2 | missing_tool | asset_monitoring → planning | Asset forwarded to planning; generic steps without a saving-plan calculation or knowledge lookup. |
| pln_07 | peer_to_peer | 3 | wrong_route | planning → recommendation | Planning forwarded a planning question to recommendation; the answer (3-6 months) was right but came from the wrong agent. |
| rec_01 | hub_spoke | 3 | wrong_route | general | Router sent it to general chat, which asked the user for income and spending. |
| rec_02 | hub_spoke | 3 | missing_tool | recommendation | Asked the user for income and spending the system already has. |
| rec_02 | peer_to_peer | 2 | wrong_route | report → recommendation → clarify | Report forwarded to recommendation, then asked back on a clear question. |
| rec_02 | single_agent | 1 | missing_tool | — | Asked the user for income and spending the system already has. |
| rec_02 | single_agent | 2 | missing_tool | — | Asked the user for income and spending the system already has. |
| rec_03 | hierarchical | 2 | missing_tool | recommendation | Asked for risk level and goals instead of using the user's data or the knowledge base. |
| rec_03 | hub_spoke | 2 | missing_tool | recommendation | Asked for risk level and goals instead of using the user's data or the knowledge base. |
| rec_03 | peer_to_peer | 3 | missing_tool | tax → recommendation → asset_monitoring → recommendation | Bounced through 4 agents, then asked for horizon and risk level. |
| rec_03 | single_agent | 2 | missing_tool | — | Asked for risk level and goals instead of using the user's data or the knowledge base. |
| rec_03 | single_agent | 3 | missing_tool | — | Asked for risk level and goals instead of using the user's data or the knowledge base. |
| rec_05 | hierarchical | 1 | missing_tool | recommendation | Asked the user questions instead of using their recorded data or the knowledge base. |
| rec_05 | hierarchical | 2 | missing_tool | recommendation | Asked the user questions instead of using their recorded data or the knowledge base. |
| rec_05 | hub_spoke | 1 | missing_tool | recommendation | Asked the user questions instead of using their recorded data or the knowledge base. |
| rec_05 | hub_spoke | 2 | missing_tool | recommendation | Asked the user questions instead of using their recorded data or the knowledge base. |
| rec_05 | hub_spoke | 3 | missing_tool | recommendation | Asked the user questions instead of using their recorded data or the knowledge base. |
| rec_05 | peer_to_peer | 3 | missing_tool | asset_monitoring → recommendation → asset_monitoring → recommendation | Bounced through 4 agents; asked the user questions instead of using their recorded data or the knowledge base. |
| rec_05 | single_agent | 1 | missing_tool | — | Generic crypto warning without using the user's data or the knowledge base. |
| rec_06 | peer_to_peer | 1 | wrong_route | planning → asset_monitoring → recommendation → asset_monitoring | Bounced through 4 agents; generic answer without tools. |
| rec_06 | peer_to_peer | 3 | wrong_route | planning | Planning answered from the knowledge base (reasonable content) but it is the wrong agent: no psychological-cue check. |
| rec_06 | single_agent | 2 | missing_tool | — | Asked for the holdings instead of reading the portfolio or addressing the fear. |
| rpt_01 | peer_to_peer | 3 | audit_fail | recommendation → report | Audit override: Fetched October correctly but told the user it was November. |
| rpt_02 | peer_to_peer | 3 | wrong_route | report → recommendation → report → recommendation | Bounced between report and recommendation; final answer is the recommendation agent's health score, not the report. |
| rpt_03 | peer_to_peer | 2 | wrong_route | tax → report → clarify | Tax forwarded to report, then asked back on a clear question. |
| rpt_04 | hierarchical | 1 | wrong_route | expense | Routed to expense; Asked the user for last month's income (already recorded). |
| rpt_04 | peer_to_peer | 1 | wrong_route | expense | Expense agent: Asked the user for last month's income (already recorded). |
| rpt_04 | peer_to_peer | 2 | wrong_route | expense | Expense agent: Asked the user for last month's income (already recorded). |
| rpt_04 | peer_to_peer | 3 | wrong_route | expense → clarify | Expense agent asked back on a clear question. |
| rpt_04 | single_agent | 1 | missing_tool | — | Asked the user for last month's income (already recorded). |
| rpt_04 | single_agent | 2 | missing_tool | — | Asked the user for last month's income (already recorded). |
| rpt_04 | single_agent | 3 | missing_tool | — | Asked the user for last month's income (already recorded). |
| rpt_05 | hub_spoke | 2 | wrong_value | report | Housing shown as 9,900 instead of 9,000 (the 12,930 total is right). |
| rpt_05 | hub_spoke | 3 | audit_fail | report | Audit override: Fetched October correctly but told the user it was November. |
| rpt_05 | single_agent | 1 | audit_fail | — | Audit override: Fetched October correctly but told the user it was September. |
| rpt_06 | hierarchical | 3 | missing_tool | report | Asked to confirm Q4 instead of answering for the current quarter. |
| rpt_06 | single_agent | 1 | missing_tool | — | Asked which quarter instead of answering for the current quarter. |
| rpt_06 | single_agent | 2 | missing_tool | — | Asked which quarter instead of answering for the current quarter. |
| rpt_06 | single_agent | 3 | missing_tool | — | Asked which quarter instead of answering for the current quarter. |
| tax_05 | hierarchical | 2 | missing_tool | tax | Asked to confirm that 35,000 is monthly instead of computing the tax. |
| tax_05 | peer_to_peer | 1 | empty_answer | recommendation → tax | Forwarded to tax; empty reply from the model. |
| tax_07 | hierarchical | 3 | empty_answer | tax | Empty reply from the model. |

## Patterns (all 3 rounds)

- **Cross-topic tasks** fail most in every design (47-58 % success). A routed design
  sends the question to one agent, and that agent often lacks the other domain's tool:
  the tax agent has no saving-plan or portfolio tool (crs_05 and crs_09 fail in
  every routed run). The single agent has every tool but still often skips the second part.
- **Peer-to-peer** loses most runs to routing (26 wrong routes). Entry agents ask back
  on clear questions or bounce a question through 3-4 agents. It is the only design
  that is significantly worse than hub-and-spoke (McNemar p = 0.041).
- **Hierarchical** misroutes mainly at level 1, where the group classifier is unsure
  and asks back on clear questions (ast_01, ast_04, ast_06, crs_11).
- **Single agent** never misroutes but most often skips a needed tool or asks the user
  for data the system already has (rpt_04, rec_02, rpt_06).
- **Empty replies** (8 runs) are a gpt-oss:20b behaviour, mostly after the tax agent
  received a two-part question.
- **Wrong month labels** (6 runs) come from the model, not the tools: the tool calls
  asked for October every time.
