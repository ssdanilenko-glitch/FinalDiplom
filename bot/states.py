"""FSM-состояния бота.

Одна StatesGroup = один сценарий. Сейчас единственный — AskFlow:
выбор темы → текст вопроса → подтверждение/отправка.
"""

from aiogram.fsm.state import State, StatesGroup


class AskFlow(StatesGroup):
    waiting_for_topic = State()
    waiting_for_question = State()
    confirming = State()
