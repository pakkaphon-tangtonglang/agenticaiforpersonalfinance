"""add risk_assessments table

Revision ID: c4d81f2a97b5
Revises: 617796925081
Create Date: 2026-04-10 09:15:42.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c4d81f2a97b5"
down_revision: Union[str, Sequence[str], None] = "617796925081"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: create risk_assessments for SEC suitability results."""
    op.create_table(
        "risk_assessments",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("answers", sa.JSON(), nullable=False),
        sa.Column("total_score", sa.Integer(), nullable=False),
        sa.Column("risk_level", sa.Integer(), nullable=False),
        sa.Column("risk_category", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_risk_assessments_user_id"), "risk_assessments", ["user_id"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema: drop the risk_assessments table and its index."""
    op.drop_index(op.f("ix_risk_assessments_user_id"), table_name="risk_assessments")
    op.drop_table("risk_assessments")
