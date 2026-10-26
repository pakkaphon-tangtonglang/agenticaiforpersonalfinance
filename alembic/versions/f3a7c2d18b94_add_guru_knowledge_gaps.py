"""add guru_knowledge_gaps table

Revision ID: f3a7c2d18b94
Revises: e5b2a9c31f60
Create Date: 2026-10-27 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "f3a7c2d18b94"
down_revision: Union[str, Sequence[str], None] = "e5b2a9c31f60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: log guru questions the knowledge base could not answer."""
    op.create_table(
        "guru_knowledge_gaps",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("guru_id", sa.String(length=50), nullable=False),
        sa.Column("query", sa.String(length=500), nullable=False),
        sa.Column("best_distance", sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_guru_knowledge_gaps_guru_id"), "guru_knowledge_gaps", ["guru_id"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema: drop guru_knowledge_gaps."""
    op.drop_index(op.f("ix_guru_knowledge_gaps_guru_id"), table_name="guru_knowledge_gaps")
    op.drop_table("guru_knowledge_gaps")
