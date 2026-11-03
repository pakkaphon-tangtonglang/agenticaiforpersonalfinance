"""Result of one task run through one architecture."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from finance_ai.evaluation.architectures.run_metrics import RunMetrics

DesignName = Literal["single_agent", "hub_spoke", "hierarchical", "peer_to_peer"]
CLARIFY_ROUTE = "clarify"


class ArchitectureRun(BaseModel):
    """What one design did with one task in one round.

    Attributes:
        task_id: Task that was run.
        design: Architecture name.
        round_index: Repeat number (0-based).
        answer: Final text shown to the user.
        route: Agents visited in order; ``["clarify"]`` when the system asked
            back; empty for the single agent (it has no routing step).
        metrics: LLM calls, tokens and tool calls inside the run.
        latency_seconds: Wall-clock time of the whole run.
        error: Exception text when the run crashed.

    Example:
        >>> run = ArchitectureRun(task_id="t", design="hub_spoke", round_index=0,
        ...                       answer="ok", route=["tax"])
        >>> run.final_agent
        'tax'
    """

    task_id: str
    design: DesignName
    round_index: int = Field(ge=0)
    answer: str = ""
    route: list[str] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    error: Optional[str] = None

    @property
    def final_agent(self) -> Optional[str]:
        """Return the agent that produced the answer, if any."""
        return self.route[-1] if self.route else None

    @property
    def asked_back_by_router(self) -> bool:
        """Return True when a routing step chose to ask the user back."""
        return self.final_agent == CLARIFY_ROUTE
