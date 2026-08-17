"""Фикстуры для bot-тестов.

Тесты бота не запускают polling и не требуют реального BOT_TOKEN —
проверяют только BackendClient (httpx.MockTransport) и FSM-стейты.
"""

import os

# BOT_TOKEN нужен для импорта config — в тестах подсовываем заглушку,
# если разработчик не задал её в окружении.
os.environ.setdefault("BOT_TOKEN", "test-token")
