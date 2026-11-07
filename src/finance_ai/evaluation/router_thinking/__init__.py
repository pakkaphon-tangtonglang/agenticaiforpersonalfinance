"""Router study (4.11 B): two Gemini models × two thinking levels on the same questions.

One run answers three questions from the same log:
- 3.5-flash minimal vs no setting: the ch.3 claim "minimal thinking, no loss in accuracy".
- flash-lite minimal vs high: does thinking help the cheaper model?
- flash-lite minimal vs 3.5-flash minimal: could flash-lite replace the production router?
"Low" is not a setup: the thinking probe showed it spends 0 thinking tokens on routing.
Notes for the thesis: docs/evaluation/router-thinking-notes.md.
"""
