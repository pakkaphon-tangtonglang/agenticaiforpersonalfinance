"""add sources to conversation_messages

Revision ID: b9e1f4a7c302
Revises: a8d4e6f20c13
Create Date: 2026-10-31 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "b9e1f4a7c302"
down_revision: Union[str, Sequence[str], None] = "a8d4e6f20c13"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: store citation source cards with assistant messages."""
    op.add_column("conversation_messages", sa.Column("sources", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema: drop the citation sources column."""
    op.drop_column("conversation_messages", "sources")
