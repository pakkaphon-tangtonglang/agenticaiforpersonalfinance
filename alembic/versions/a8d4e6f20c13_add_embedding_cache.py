"""add embedding_cache table

Revision ID: a8d4e6f20c13
Revises: f3a7c2d18b94
Create Date: 2026-10-27 18:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "a8d4e6f20c13"
down_revision: Union[str, Sequence[str], None] = "f3a7c2d18b94"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: store embedding vectors so restarts reuse them."""
    op.create_table(
        "embedding_cache",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_embedding_cache_cache_key"), "embedding_cache", ["cache_key"], unique=True
    )


def downgrade() -> None:
    """Downgrade schema: drop embedding_cache."""
    op.drop_index(op.f("ix_embedding_cache_cache_key"), table_name="embedding_cache")
    op.drop_table("embedding_cache")
