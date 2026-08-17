"""Модерация входящих сообщений — упрощённый каскад regex → OpenAI Moderation.

Цель: блокировать явно вредный контент ДО сохранения в БД и отправки в LLM.
"""
from app.moderation.domain import ModerationResult
from app.moderation.exceptions import ModerationBlockedError
from app.moderation.service import ModerationService

__all__ = ["ModerationResult", "ModerationBlockedError", "ModerationService"]
