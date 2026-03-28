"""add asset schedule and notification tables

Revision ID: a3f1c8d92e47
Revises: 8075b8aa76a8
Create Date: 2026-03-28 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a3f1c8d92e47"
down_revision: Union[str, Sequence[str], None] = "8075b8aa76a8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create asset_schedules and asset_notifications tables."""
    op.create_table(
        "asset_schedules",
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=False),
        sa.Column("cron_expression", sa.String(length=100), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_asset_schedules_user_id"),
        "asset_schedules",
        ["user_id"],
    )

    op.create_table(
        "asset_notifications",
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("schedule_id", sa.String(length=36), nullable=True),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("is_read", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["schedule_id"], ["asset_schedules.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_asset_notifications_user_id"),
        "asset_notifications",
        ["user_id"],
    )


def downgrade() -> None:
    """Drop asset_notifications and asset_schedules tables."""
    op.drop_index(
        op.f("ix_asset_notifications_user_id"),
        table_name="asset_notifications",
    )
    op.drop_table("asset_notifications")
    op.drop_index(
        op.f("ix_asset_schedules_user_id"),
        table_name="asset_schedules",
    )
    op.drop_table("asset_schedules")
