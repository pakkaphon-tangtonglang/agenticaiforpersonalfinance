"""add debts table and link installments to a debt

Revision ID: e5b1c7d93a20
Revises: d2f6b8a41c37
Create Date: 2026-11-04 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "e5b1c7d93a20"
down_revision: Union[str, Sequence[str], None] = "d2f6b8a41c37"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "debts",
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("creditor", sa.String(length=255), nullable=False),
        sa.Column("debt_type", sa.String(length=30), nullable=False),
        sa.Column("original_amount", sa.Numeric(precision=16, scale=2), nullable=False),
        sa.Column("remaining_balance", sa.Numeric(precision=16, scale=2), nullable=False),
        sa.Column("interest_rate_percent", sa.Numeric(precision=7, scale=4), nullable=True),
        sa.Column("monthly_payment", sa.Numeric(precision=16, scale=2), nullable=True),
        sa.Column("is_paid_off", sa.Boolean(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_debts_user_id"), "debts", ["user_id"], unique=False)
    with op.batch_alter_table("recurring_entries") as batch_op:
        batch_op.add_column(sa.Column("debt_id", sa.String(length=36), nullable=True))
        batch_op.create_foreign_key(
            "fk_recurring_entries_debt_id_debts", "debts", ["debt_id"], ["id"]
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("recurring_entries") as batch_op:
        batch_op.drop_constraint("fk_recurring_entries_debt_id_debts", type_="foreignkey")
        batch_op.drop_column("debt_id")
    op.drop_index(op.f("ix_debts_user_id"), table_name="debts")
    op.drop_table("debts")
