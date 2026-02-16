"""Tests for logging configuration."""

import logging
from finance_ai.core.logging import configure_logging, get_logger


def test_configure_logging_default_level() -> None:
    """Test configure_logging sets default level."""
    configure_logging()

    logger = logging.getLogger("finance_ai")
    assert logger.level == logging.INFO


def test_configure_logging_custom_level() -> None:
    """Test configure_logging sets custom level."""
    configure_logging(level="DEBUG")

    logger = logging.getLogger("finance_ai")
    assert logger.level == logging.DEBUG


def test_get_logger_returns_logger() -> None:
    """Test get_logger returns proper logger instance."""
    logger = get_logger("test_module")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_get_logger_inherits_configuration() -> None:
    """Test get_logger inherits parent configuration."""
    configure_logging(level="WARNING")
    logger = get_logger("finance_ai.test")

    assert logger.level == logging.NOTSET
    assert logger.parent is not None
    assert logger.parent.level == logging.WARNING
