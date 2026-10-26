"""add conversation kind and guru_id

Revision ID: e5b2a9c31f60
Revises: b7e3d5c8a214
Create Date: 2026-10-27 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "e5b2a9c31f60"
down_revision: Union[str, Sequence[str], None] = "b7e3d5c8a214"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: separate Guru tab chats from main chats."""
    with op.batch_alter_table("conversations") as batch_op:
        batch_op.add_column(
            sa.Column("kind", sa.String(length=20), nullable=False, server_default="chat")
        )
        batch_op.add_column(sa.Column("guru_id", sa.String(length=50), nullable=True))


def downgrade() -> None:
    """Downgrade schema: drop the Guru tab columns."""
    with op.batch_alter_table("conversations") as batch_op:
        batch_op.drop_column("guru_id")
        batch_op.drop_column("kind")
