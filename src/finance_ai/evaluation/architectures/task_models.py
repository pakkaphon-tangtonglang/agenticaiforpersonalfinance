"""Task definitions for the architecture comparison study.

A task is one user question plus what a correct run must show: which
agent should take it, which tools must be called, which values must appear
in the answer and which rows must be written to the database.
"""

from decimal import Decimal
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

AgentName = Literal[
    "tax", "expense", "asset_monitoring", "planning", "recommendation", "report", "general"
]
TaskCategory = Literal["single", "cross", "vague"]
DbCheckKind = Literal["expense", "income", "goal_target", "goal_current", "watchlist", "holding"]

DEFAULT_TASKS_PATH = Path("data/evaluation/architecture_tasks.yaml")


class DbCheck(BaseModel):
    """A row the run must have written to the database.

    Example:
        >>> DbCheck(kind="expense", amount=Decimal("80")).kind
        'expense'
    """

    kind: DbCheckKind
    amount: Optional[Decimal] = None
    symbol: Optional[str] = None


class ArchitectureTask(BaseModel):
    """One study task.

    Attributes:
        task_id: Unique id, e.g. ``tax_01``.
        query: Thai user question.
        category: single topic, cross-topic, or vague (should ask back).
        acceptable_agents: Agents that may answer; empty for vague tasks.
        required_tools: Groups of tool names; each group needs one call.
        answer_checks: Values that must appear in the answer; ``|`` separates
            accepted alternatives; commas are ignored when matching.
        db_checks: Rows the run must write.

    Example:
        >>> ArchitectureTask(task_id="t", query="q", category="vague").expects_clarify
        True
    """

    task_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    category: TaskCategory
    acceptable_agents: list[AgentName] = Field(default_factory=list)
    required_tools: list[list[str]] = Field(default_factory=list)
    answer_checks: list[str] = Field(default_factory=list)
    db_checks: list[DbCheck] = Field(default_factory=list)
    note: str = ""

    @property
    def expects_clarify(self) -> bool:
        """Return True when the right outcome is asking the user back."""
        return self.category == "vague"

    @model_validator(mode="after")
    def _check_category_rules(self) -> "ArchitectureTask":
        """Vague tasks route nowhere; other tasks must name an agent."""
        if self.expects_clarify and self.acceptable_agents:
            raise ValueError(f"Vague task {self.task_id} must not list acceptable_agents.")
        if not self.expects_clarify and not self.acceptable_agents:
            raise ValueError(f"Task {self.task_id} needs at least one acceptable agent.")
        return self


class ArchitectureTaskSet(BaseModel):
    """The full task file."""

    name: str
    version: str
    tasks: list[ArchitectureTask]

    @model_validator(mode="after")
    def _check_unique_ids(self) -> "ArchitectureTaskSet":
        """Reject duplicate task ids."""
        ids = [task.task_id for task in self.tasks]
        duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
        if duplicates:
            raise ValueError(f"Duplicate task ids: {duplicates}")
        return self


def load_architecture_tasks(path: Path = DEFAULT_TASKS_PATH) -> ArchitectureTaskSet:
    """Load and validate the task file.

    Args:
        path: YAML file with ``name``, ``version`` and ``tasks``.

    Returns:
        Validated task set.

    Raises:
        FileNotFoundError: If the file does not exist.

    Example:
        >>> task_set = load_architecture_tasks()
        >>> len(task_set.tasks)
        60
    """
    if not path.exists():
        raise FileNotFoundError(f"Architecture task file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ArchitectureTaskSet.model_validate(raw)
