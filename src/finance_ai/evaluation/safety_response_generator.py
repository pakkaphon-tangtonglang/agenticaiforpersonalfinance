"""Generates Recommendation Agent responses for the safety evaluation.

Runs each safety case through the real Recommendation Agent graph (not the
router, so a crypto query cannot escape to asset monitoring) with a simulated
user seeded at the case's risk level. The post-generation guardrail therefore
executes exactly as in production and its warnings land in the final answer.
"""

import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from finance_ai.core.logging import get_logger
from finance_ai.evaluation.llm_retry import invoke_with_retry

from finance_ai.agents.recommendation_agent import build_recommendation_agent_graph
from finance_ai.core.logging import get_logger
from finance_ai.database.base import Base
from finance_ai.database.models.risk_assessment import RiskAssessment
from finance_ai.database.models.user import User
from finance_ai.evaluation.models import RecommendationSafetyCase, RecommendationSafetyDataset
from finance_ai.tools.risk_assessment_constants import RISK_LEVELS

logger = get_logger(__name__)

_SAFETY_EVAL_NAMESPACE = uuid.UUID("b16a5c2e-0000-4000-8000-000000000001")


def create_isolated_session_factory() -> Callable[[], Session]:
    """Create an in-memory SQLite factory with all tables for eval runs.

    Uses StaticPool so every session (including agent tool sessions)
    shares the same in-memory database, mirroring the test setup.

    Returns:
        Session factory bound to an isolated in-memory database.

    Example:
        >>> factory = create_isolated_session_factory()
    """
    engine = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


def _risk_level_band(risk_level: int) -> tuple[int, str]:
    """Return a representative score and Thai category for a risk level.

    Args:
        risk_level: Questionnaire risk level (1-5).

    Returns:
        Tuple of (total_score, risk_category).

    Raises:
        ValueError: If the risk level has no band definition.

    Example:
        >>> _risk_level_band(1)
        (10, 'เสี่ยงต่ำ')
    """
    for min_score, _max_score, level, category in RISK_LEVELS:
        if level == risk_level:
            return min_score, category
    raise ValueError(f"Unknown risk level: {risk_level}")


def seed_safety_user(
    db_session_factory: Callable[[], Session],
    case_id: str,
    risk_level: int,
) -> str:
    """Create a simulated user with a risk assessment at the given level.

    Args:
        db_session_factory: Factory for the isolated eval database.
        case_id: Case id driving the deterministic user UUID.
        risk_level: Questionnaire risk level (1-5) to seed.

    Returns:
        The seeded user's UUID string.

    Example:
        >>> user_id = seed_safety_user(factory, "safety_001", 1)
    """
    user_id = _safety_user_id(case_id)
    total_score, category = _risk_level_band(risk_level)
    with db_session_factory() as session:
        if session.get(User, user_id) is None:
            session.add(_build_safety_user(user_id, case_id))
        session.add(_build_safety_assessment(user_id, case_id, total_score, risk_level, category))
        session.commit()
    return user_id


def _safety_user_id(case_id: str) -> str:
    """Build the deterministic user UUID for a safety case.

    Args:
        case_id: Safety case id.

    Returns:
        UUID string derived from the case id.

    Example:
        >>> _safety_user_id("safety_001")
        '...uuid5...'
    """
    return str(uuid.uuid5(_SAFETY_EVAL_NAMESPACE, f"safety-eval-{case_id}"))


def _build_safety_user(user_id: str, case_id: str) -> User:
    """Build the simulated User row for a safety case.

    Args:
        user_id: Pre-computed deterministic UUID.
        case_id: Safety case id for the display name.

    Returns:
        A User instance ready to add.
    """
    return User(
        id=user_id,
        email=f"{user_id}@safety.eval",
        hashed_password="!",
        full_name=f"Safety Eval {case_id}",
    )


def _build_safety_assessment(
    user_id: str,
    case_id: str,
    total_score: int,
    risk_level: int,
    category: str,
) -> RiskAssessment:
    """Build the RiskAssessment row giving the user the case's risk level.

    Args:
        user_id: Simulated user's UUID.
        case_id: Safety case id (kept in answers for traceability).
        total_score: Representative questionnaire score.
        risk_level: Questionnaire risk level (1-5).
        category: Thai risk category label.

    Returns:
        A RiskAssessment instance ready to add.
    """
    return RiskAssessment(
        user_id=user_id,
        answers={"case_id": case_id},
        total_score=total_score,
        risk_level=risk_level,
        risk_category=category,
    )


def generate_safety_responses(
    chat_model: BaseChatModel,
    dataset: RecommendationSafetyDataset,
    db_session_factory: Callable[[], Session] | None = None,
) -> tuple[dict[str, str | None], dict[str, bool]]:
    """Run every safety case through the Recommendation Agent graph.

    Args:
        chat_model: Model being evaluated.
        dataset: Safety evaluation dataset.
        db_session_factory: Optional factory; an isolated in-memory
            database is created when None.

    Returns:
        Tuple of (responses by case_id, had_tool_results by case_id).
        A response is None when its generation failed.

    Example:
        >>> responses, flags = generate_safety_responses(model, dataset)
    """
    factory = db_session_factory or create_isolated_session_factory()
    graph = build_recommendation_agent_graph(chat_model)
    responses: dict[str, str | None] = {}
    tool_flags: dict[str, bool] = {}
    for case in dataset.cases:
        response, had_tools = _invoke_graph_for_case(graph, factory, case)
        responses[case.case_id] = response
        tool_flags[case.case_id] = had_tools
    return responses, tool_flags


def _invoke_graph_for_case(
    graph: Any,
    db_session_factory: Callable[[], Session],
    case: RecommendationSafetyCase,
) -> tuple[str | None, bool]:
    """Invoke the graph for one case and extract the final answer.

    Args:
        graph: Compiled Recommendation Agent graph.
        db_session_factory: Session factory for the isolated eval database.
        case: Safety case to run.

    Returns:
        Tuple of (final answer text, whether tool results were used).
        The response is None when generation failed entirely, so the
        evaluator records a missing_response failure instead of letting
        an empty string pass the guardrail trivially.
    """
    user_id = seed_safety_user(db_session_factory, case.case_id, case.risk_level)
    try:
        result = invoke_with_retry(
            lambda: graph.invoke(
                {
                    "messages": [("user", case.query)],
                    "user_id": user_id,
                    "db_session_factory": db_session_factory,
                }
            )
        )
    except Exception as error:  # noqa: BLE001  # one bad case must not kill the run
        logger.warning("Safety case %s failed: %s", case.case_id, error)
        return None, False
    messages = list(result.get("messages", []))
    return _extract_answer(messages), any(isinstance(m, ToolMessage) for m in messages)


def _extract_answer(messages: list[Any]) -> str:
    """Return the last message content as text ('' when absent).

    Args:
        messages: Final message list from the graph.

    Returns:
        The final answer text.
    """
    if not messages:
        return ""
    return str(messages[-1].content)
