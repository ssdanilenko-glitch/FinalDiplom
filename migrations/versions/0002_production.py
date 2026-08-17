"""production tables: rate-limit, feedback, prompts, alerts, broadcasts

Добавляем таблицы для production-обвязки:
- rate_limits — атомарный счётчик rate-limit (PRIMARY KEY как UNIQUE для UPSERT)
- message_feedback — реакции пользователей (up/down) на ответы ассистента
- system_prompts — версионированные системные промпты с A/B traffic split
- alerts — очередь алертов для drain'а в Telegram админ-чат
- broadcasts — статус массовых рассылок
- chats.handoff_status — флаг передачи диалога оператору
- chat_messages.prompt_id FK → system_prompts.id (mapping prompt → message)

Revision ID: 0002_production
Revises: 0001_chats
Create Date: 2026-05-21
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0002_production"
down_revision: Union[str, None] = "0001_chats"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- system_prompts (создаём ДО FK из chat_messages) -----------------
    op.create_table(
        "system_prompts",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
        sa.Column(
            "traffic_pct",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("notes", sa.Text(), nullable=True),
    )

    # --- chat_messages.prompt_id FK --------------------------------------
    op.create_foreign_key(
        "fk_chat_messages_prompt_id",
        source_table="chat_messages",
        referent_table="system_prompts",
        local_cols=["prompt_id"],
        remote_cols=["id"],
        ondelete="SET NULL",
    )

    # --- chats.handoff_status --------------------------------------------
    op.add_column(
        "chats",
        sa.Column(
            "handoff_status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
    )

    # --- rate_limits ------------------------------------------------------
    op.create_table(
        "rate_limits",
        sa.Column("owner_external_id", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("bucket", sa.Text(), nullable=False),
        sa.Column(
            "count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.PrimaryKeyConstraint(
            "owner_external_id",
            "kind",
            "bucket",
            name="pk_rate_limits",
        ),
    )
    op.create_index(
        "idx_rate_limits_bucket",
        "rate_limits",
        ["bucket"],
    )

    # --- message_feedback -------------------------------------------------
    op.create_table(
        "message_feedback",
        sa.Column("message_id", UUID(as_uuid=True), nullable=False),
        sa.Column("owner_external_id", sa.Text(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint(
            "owner_external_id",
            "message_id",
            name="pk_message_feedback",
        ),
    )

    # --- alerts -----------------------------------------------------------
    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column(
            "payload",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("acked_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )

    # --- broadcasts -------------------------------------------------------
    op.create_table(
        "broadcasts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("total_owners", sa.Integer(), nullable=False),
        sa.Column(
            "sent", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "failed", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("last_owner_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("broadcasts")
    op.drop_table("alerts")
    op.drop_table("message_feedback")
    op.drop_index("idx_rate_limits_bucket", table_name="rate_limits")
    op.drop_table("rate_limits")
    op.drop_column("chats", "handoff_status")
    op.drop_constraint(
        "fk_chat_messages_prompt_id", "chat_messages", type_="foreignkey"
    )
    op.drop_table("system_prompts")
