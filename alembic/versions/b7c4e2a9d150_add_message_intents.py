"""add intents to conversation messages (multi-topic replies)

Revision ID: b7c4e2a9d150
Revises: e5b1c7d93a20
Create Date: 2026-11-07 18:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "b7c4e2a9d150"
down_revision: Union[str, Sequence[str], None] = "e5b1c7d93a20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("conversation_messages", sa.Column("intents", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("conversation_messages", "intents")
