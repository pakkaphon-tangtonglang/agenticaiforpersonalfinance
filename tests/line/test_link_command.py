"""Tests for the LINE -> web account link command."""

from sqlalchemy.orm import Session

from finance_ai.database.models.line_user_mapping import LineUserMapping
from finance_ai.database.models.user import User
from finance_ai.line.link_command import (
    link_line_user,
    parse_link_command,
    parse_unlink_command,
    unlink_line_user,
)
from finance_ai.line.mapping_service import get_or_create_line_mapping

LINE_USER_ID = "Uline-user-link"
TARGET_USER_ID = "00000000-de20-4000-8000-000000000001"


class TestParseLinkCommand:
    """Tests for parse_link_command."""

    def test_parses_thai_prefix(self) -> None:
        """'เชื่อมต่อ <id>' yields the target user id."""
        assert parse_link_command(f"เชื่อมต่อ {TARGET_USER_ID}") == TARGET_USER_ID

    def test_parses_english_prefix_case_insensitive(self) -> None:
        """'LINK <id>' also works, case-insensitively."""
        assert parse_link_command(f"LINK {TARGET_USER_ID}") == TARGET_USER_ID

    def test_ignores_extra_whitespace(self) -> None:
        """Multiple spaces between prefix and id are tolerated."""
        assert parse_link_command(f"เชื่อมต่อ   {TARGET_USER_ID}") == TARGET_USER_ID

    def test_non_command_returns_none(self) -> None:
        """Normal chat messages are not link commands."""
        assert parse_link_command("ภาษีของฉันเท่าไหร่") is None

    def test_prefix_without_id_returns_none(self) -> None:
        """The bare command without an id is not actionable."""
        assert parse_link_command("เชื่อมต่อ") is None


class TestLinkLineUser:
    """Tests for link_line_user."""

    def _seed_mapping(self, session: Session) -> LineUserMapping:
        """Create the auto-generated LINE mapping."""
        return get_or_create_line_mapping(session, LINE_USER_ID)

    def _seed_target_user(self, session: Session) -> None:
        """Create the web-app user the LINE account will link to."""
        session.add(
            User(
                id=TARGET_USER_ID,
                email="web@finance-ai.local",
                hashed_password="not-a-login",
                full_name="ผู้ใช้เว็บ",
            )
        )
        session.commit()

    def test_repoints_mapping_to_existing_user(self, test_session: Session) -> None:
        """A valid link updates the mapping's user_id."""
        self._seed_mapping(test_session)
        self._seed_target_user(test_session)
        success, message = link_line_user(test_session, LINE_USER_ID, TARGET_USER_ID)
        assert success is True
        mapping = test_session.query(LineUserMapping).filter_by(line_user_id=LINE_USER_ID).one()
        assert mapping.user_id == TARGET_USER_ID

    def test_unknown_user_rejected(self, test_session: Session) -> None:
        """An id that does not exist is rejected without changing the mapping."""
        self._seed_mapping(test_session)
        success, message = link_line_user(
            test_session, LINE_USER_ID, "ffffffff-0000-4000-8000-000000000000"
        )
        assert success is False
        assert "ไม่พบ" in message
        mapping = test_session.query(LineUserMapping).filter_by(line_user_id=LINE_USER_ID).one()
        assert mapping.user_id != "ffffffff-0000-4000-8000-000000000000"

    def test_unlinked_line_user_cannot_link(self, test_session: Session) -> None:
        """A LINE user with no mapping gets an error (mapping is created lazily)."""
        success, message = link_line_user(test_session, "Unever-seen-before", TARGET_USER_ID)
        assert success is False
        assert "ไม่พบ" in message


class TestParseUnlinkCommand:
    """Tests for parse_unlink_command."""

    def test_thai_command_matches(self) -> None:
        """'ยกเลิกเชื่อมต่อ' is an unlink command."""
        assert parse_unlink_command("ยกเลิกเชื่อมต่อ") is True

    def test_english_command_case_insensitive(self) -> None:
        """'UNLINK' and 'unlink' are unlink commands."""
        assert parse_unlink_command("UNLINK") is True
        assert parse_unlink_command("unlink") is True

    def test_normal_text_is_not_unlink(self) -> None:
        """Chat messages and link commands are not unlink commands."""
        assert parse_unlink_command("ภาษีของฉันเท่าไหร่") is False
        assert parse_unlink_command(f"เชื่อมต่อ {TARGET_USER_ID}") is False


class TestUnlinkLineUser:
    """Tests for unlink_line_user."""

    def _seed_mapping(self, session: Session) -> LineUserMapping:
        """Create the auto-generated LINE mapping."""
        return get_or_create_line_mapping(session, LINE_USER_ID)

    def test_restores_original_line_user(self, test_session: Session) -> None:
        """After unlink, the mapping points back at the auto-created user."""
        mapping = self._seed_mapping(test_session)
        original_user_id = mapping.user_id

        test_session.add(
            User(
                id=TARGET_USER_ID,
                email="web@finance-ai.local",
                hashed_password="not-a-login",
                full_name="ผู้ใช้เว็บ",
            )
        )
        test_session.commit()
        link_line_user(test_session, LINE_USER_ID, TARGET_USER_ID)
        success, message = unlink_line_user(test_session, LINE_USER_ID)

        assert success is True
        assert "ยกเลิก" in message
        mapping = test_session.query(LineUserMapping).filter_by(line_user_id=LINE_USER_ID).one()
        assert mapping.user_id == original_user_id

    def test_without_mapping_replies_error(self, test_session: Session) -> None:
        """A LINE user that never chatted cannot unlink."""
        success, message = unlink_line_user(test_session, "Unever-seen")
        assert success is False
        assert "ยังไม่มีการเชื่อมต่อ" in message
