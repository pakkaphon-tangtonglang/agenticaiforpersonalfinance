"""Tests for the agent session helper."""

from unittest.mock import MagicMock, patch

from finance_ai.agents.session_helper import get_tool_session


class TestGetToolSession:
    """Tests for the get_tool_session context manager."""

    def test_uses_provided_factory(self) -> None:
        """Uses the provided factory to create a session."""
        mock_session = MagicMock()
        factory = MagicMock(return_value=mock_session)

        with get_tool_session(factory) as session:
            assert session is mock_session

        factory.assert_called_once()
        mock_session.close.assert_called_once()

    def test_closes_session_on_exception(self) -> None:
        """Closes session even when exception occurs."""
        mock_session = MagicMock()
        factory = MagicMock(return_value=mock_session)

        try:
            with get_tool_session(factory) as session:
                raise ValueError("test error")
        except ValueError:
            pass

        mock_session.close.assert_called_once()

    @patch("finance_ai.database.session.get_database_session")
    def test_uses_default_when_no_factory(
        self,
        mock_get_session: MagicMock,
    ) -> None:
        """Falls back to default database session when factory is None."""
        mock_session = MagicMock()
        mock_get_session.return_value = iter([mock_session])

        with get_tool_session(None) as session:
            assert session is mock_session

    def test_none_factory_triggers_default(self) -> None:
        """Passing None explicitly triggers the default path."""
        with patch("finance_ai.database.session.get_database_session") as mock_get:
            mock_session = MagicMock()
            mock_get.return_value = iter([mock_session])

            with get_tool_session() as session:
                assert session is mock_session
