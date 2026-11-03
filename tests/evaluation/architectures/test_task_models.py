"""Tests for the architecture study task models and the real task file."""

from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from finance_ai.evaluation.architectures.task_models import (
    ArchitectureTask,
    ArchitectureTaskSet,
    load_architecture_tasks,
)
from finance_ai.evaluation.architectures.single_agent import all_tools
from finance_ai.tools.tax_calculator import calculate_tax


def _all_agent_tool_names() -> set[str]:
    """Return every tool name the six specialist agents can call."""
    return {tool.name for tool in all_tools()}


class TestArchitectureTask:
    """Validation rules on a single task."""

    def test_vague_task_expects_clarify(self) -> None:
        """A vague task expects the system to ask back."""
        task = ArchitectureTask(task_id="v1", query="เงิน", category="vague")
        assert task.expects_clarify

    def test_vague_task_cannot_name_agents(self) -> None:
        """A vague task with an agent is rejected."""
        with pytest.raises(ValidationError, match="must not list"):
            ArchitectureTask(
                task_id="v1", query="เงิน", category="vague", acceptable_agents=["tax"]
            )

    def test_single_task_needs_an_agent(self) -> None:
        """A non-vague task without an agent is rejected."""
        with pytest.raises(ValidationError, match="needs at least one"):
            ArchitectureTask(task_id="t1", query="ภาษี", category="single")

    def test_duplicate_ids_rejected(self) -> None:
        """Two tasks with the same id are rejected."""
        task = ArchitectureTask(task_id="t1", query="q", category="vague")
        with pytest.raises(ValidationError, match="Duplicate"):
            ArchitectureTaskSet(name="n", version="1", tasks=[task, task])

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A missing task file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_architecture_tasks(tmp_path / "none.yaml")


@pytest.fixture(scope="module", name="task_set")
def fixture_task_set() -> ArchitectureTaskSet:
    """Load the real task file once."""
    return load_architecture_tasks()


class TestRealTaskFile:
    """The committed task file matches the study design."""

    def test_has_sixty_tasks_in_planned_mix(self, task_set: ArchitectureTaskSet) -> None:
        """40 single-topic, 12 cross-topic and 8 vague tasks."""
        counts = Counter(task.category for task in task_set.tasks)
        assert counts == {"single": 40, "cross": 12, "vague": 8}

    def test_single_topic_spread_over_agents(self, task_set: ArchitectureTaskSet) -> None:
        """Single-topic tasks follow the planned per-agent counts."""
        singles = [t.acceptable_agents[0] for t in task_set.tasks if t.category == "single"]
        assert Counter(singles) == {
            "tax": 7,
            "expense": 7,
            "asset_monitoring": 7,
            "planning": 7,
            "recommendation": 6,
            "report": 6,
        }

    def test_cross_tasks_need_two_tool_groups(self, task_set: ArchitectureTaskSet) -> None:
        """Every cross-topic task needs at least two tool groups."""
        crosses = [t for t in task_set.tasks if t.category == "cross"]
        assert all(len(task.required_tools) >= 2 for task in crosses)

    def test_required_tools_exist(self, task_set: ArchitectureTaskSet) -> None:
        """Every required tool name is a real agent tool."""
        known = _all_agent_tool_names()
        named = {name for task in task_set.tasks for group in task.required_tools for name in group}
        assert named <= known, f"Unknown tools: {sorted(named - known)}"

    def test_tax_answers_match_the_calculator(self, task_set: ArchitectureTaskSet) -> None:
        """Expected tax figures equal the production tax calculator."""
        expected = {
            "tax_01": (600000, {"personal_allowance": 60000}),
            "tax_02": (1200000, {"personal_allowance": 60000, "child_allowance": 30000}),
            "tax_03": (960000, {"personal_allowance": 60000, "life_insurance": 50000}),
            "tax_04": (720000, {"personal_allowance": 60000, "rmf": 100000}),
            "tax_05": (420000, {"personal_allowance": 60000}),
            "tax_06": (
                2000000,
                {"personal_allowance": 60000, "parent_allowance": 30000, "health_insurance": 20000},
            ),
        }
        by_id = {task.task_id: task for task in task_set.tasks}
        for task_id, (income, deductions) in expected.items():
            result = calculate_tax(
                Decimal(income), {kind: Decimal(value) for kind, value in deductions.items()}
            )
            assert f"{result.total_tax:,.0f}" in by_id[task_id].answer_checks[0], task_id
