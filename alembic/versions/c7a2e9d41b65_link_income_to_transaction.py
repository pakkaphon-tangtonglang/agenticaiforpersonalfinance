"""link income tax copies to their transaction

Revision ID: c7a2e9d41b65
Revises: b9e1f4a7c302
Create Date: 2026-11-01 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "c7a2e9d41b65"
down_revision: Union[str, Sequence[str], None] = "b9e1f4a7c302"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: incomes.transaction_id -> transactions.id (nullable, cascade delete)."""
    with op.batch_alter_table("incomes") as batch_op:
        batch_op.add_column(sa.Column("transaction_id", sa.String(length=36), nullable=True))
        batch_op.create_index("ix_incomes_transaction_id", ["transaction_id"])
        batch_op.create_foreign_key(
            "fk_incomes_transaction_id",
            "transactions",
            ["transaction_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    """Downgrade schema: drop the income -> transaction link."""
    with op.batch_alter_table("incomes") as batch_op:
        batch_op.drop_constraint("fk_incomes_transaction_id", type_="foreignkey")
        batch_op.drop_index("ix_incomes_transaction_id")
        batch_op.drop_column("transaction_id")
