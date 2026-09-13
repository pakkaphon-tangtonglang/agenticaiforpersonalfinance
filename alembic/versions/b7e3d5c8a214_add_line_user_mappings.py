"""add line_user_mappings table

Revision ID: b7e3d5c8a214
Revises: c4d81f2a97b5
Create Date: 2026-04-12 09:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b7e3d5c8a214"
down_revision: Union[str, Sequence[str], None] = "c4d81f2a97b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: create line_user_mappings for the LINE chatbot."""
    op.create_table(
        "line_user_mappings",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("line_user_id", sa.String(length=50), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("conversation_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("line_user_id"),
    )
    op.create_index("ix_line_user_mappings_line_user_id", "line_user_mappings", ["line_user_id"])
    op.create_index("ix_line_user_mappings_user_id", "line_user_mappings", ["user_id"])


def downgrade() -> None:
    """Downgrade schema: drop line_user_mappings."""
    op.drop_index("ix_line_user_mappings_user_id", table_name="line_user_mappings")
    op.drop_index("ix_line_user_mappings_line_user_id", table_name="line_user_mappings")
    op.drop_table("line_user_mappings")
