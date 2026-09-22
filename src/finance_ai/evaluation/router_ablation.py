"""Router feature ablation experiment runner.

Runs the router's ``classify_query`` over a grid of ablation variants,
models, and rounds, logging one JSONL row per call. The log is the
single source of truth: aggregation reads it back, and restarts skip
rows that already completed.

Example:
    >>> from finance_ai.evaluation.router_ablation import ABLATION_VARIANTS
    >>> sorted(ABLATION_VARIANTS)
    ['+few_shot', '+history', '+symbol', '+threshold', 'baseline', 'full']
"""

from dataclasses import replace

from finance_ai.agents.router_agent import (
    DEFAULT_ABLATION_CONFIG,
    RouterAblationConfig,
)

# One-at-a-time variants: each turns on exactly one helper over the
# baseline; "full" reproduces the production default config exactly.
_BASELINE = RouterAblationConfig(
    include_chat_history=False,
    include_few_shot_examples=False,
    resolve_asset_hint=False,
    apply_confidence_threshold=False,
)

ABLATION_VARIANTS: dict[str, RouterAblationConfig] = {
    "baseline": _BASELINE,
    "+few_shot": replace(_BASELINE, include_few_shot_examples=True),
    "+history": replace(_BASELINE, include_chat_history=True),
    "+symbol": replace(_BASELINE, resolve_asset_hint=True),
    "+threshold": replace(_BASELINE, apply_confidence_threshold=True),
    "full": DEFAULT_ABLATION_CONFIG,
}


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse a "provider:model" string into its two parts.

    Args:
        spec: Model specification, e.g. "ollama:minimax-m3".

    Returns:
        (provider, model_name) tuple.

    Raises:
        ValueError: If the spec is not "provider:model" with both parts
            non-empty.

    Example:
        >>> parse_model_spec("ollama:minimax-m3")
        ('ollama', 'minimax-m3')
    """
    provider, separator, model_name = spec.partition(":")
    if not separator or not provider or not model_name:
        raise ValueError(f"Invalid model spec '{spec}'. Expected 'provider:model'.")
    return provider, model_name
