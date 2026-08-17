# m6_b4 — Персистентный агент: чекпоинтер, человек в цикле, стриминг, трейсинг

Тот же `llm-service`, что рос `m3_b4 → m4_b4 → m5_b2 → m5_b3 → m5_b5 → m5_b6 →
m6_b3`, на чекпоинте продвинутых паттернов LangGraph. К ReAct-агенту из m6_b3
добавлены: **персистентность** (чекпоинтер — состояние переживает рестарт),
**человек в цикле** (`interrupt` / `Command(resume=...)`) на опасном действии,
**SSE-стриминг** прогресса, демо путешествия во времени и **трейсинг агента** в
Phoenix. Ядро RAG, чат и модерация — те же, что в m6_b3.

## Что нового на этом чекпоинте

| Что добавилось | Файл | Зачем |
|---|---|---|
| Персистентный агент + HIL | `app/services/agent_persistent.py` | `build_agent(checkpointer, …)` + `agent_lifespan()` (переключатель backend); опасный `send_email` проходит через `prepare_email` → `confirm_and_send` (interrupt, реальная отправка только после resume; роль `full` пропускает подтверждение) |
| Изоляция checkpoint-таблиц | `migrations/env.py` | `include_name` — Alembic `autogenerate` не трогает `checkpoint*` (их схему ведёт `setup()`) |
| Подключение чекпоинтера | `app/main.py` | сборка агента через `AsyncExitStack` в `lifespan` — чекпоинтер держит соединение всё время работы, `setup()` один раз на старте |
| Ручки агента | `app/routers/agent.py` | `/agent/chat` (распознаёт `interrupt`), `/agent/resume`, `/agent/stream` (SSE через `astream(stream_mode=["updates", "messages"])`) |
| Трейсинг агента | `app/observability/tracing.py` | `LangChainInstrumentor` — узлы графа, вызовы инструментов и LLM попадают в спаны Phoenix (в дополнение к LlamaIndex для RAG) |
| Демо time-travel | `scripts/time_travel_demo.py` | interrupt → история чек-пойнтов → чтение прошлого состояния → две ветки из одного входа |
| Тесты | `tests/test_agent_persistent.py` | 4 теста на `AsyncSqliteSaver(":memory:")`: interrupt, approve, reject, роль `full` |

Подключение: `AGENT_CHECKPOINTER` выбирает backend (`sqlite` локально, `postgres`
в compose — чекпоинты ложатся в те же таблицы БД). Источник правды разговора —
доменная таблица (`ChatMessageRow`, с модерацией и soft-delete), а чекпоинтер —
операционная память прогона агента. `send_email` в референсе логирует отправку;
реальную интеграцию (SMTP/API) подключают в `_send_email` в `lifespan`.

## Куда смотреть

Главное в этом снимке — `app/services/agent_persistent.py` (HIL-поток и
переключатель backend) и `migrations/env.py` (`include_name`). Ручки —
`app/routers/agent.py`, подключение — `app/main.py` (`lifespan` + `AsyncExitStack`).
Всё остальное — то же ядро RAG + агент, что в m6_b3.

## Карта сервиса (что выросло)

```
app/
├── services/
│   └── agent_persistent.py   # NEW: build_agent + agent_lifespan + HIL send_email
├── routers/
│   └── agent.py              # +/agent/resume, +/agent/stream (SSE)
├── observability/
│   └── tracing.py            # +LangChainInstrumentor (трейс графа/тулзов/LLM)
├── core/config.py            # +agent_checkpointer / agent_sqlite_path
└── main.py                   # чекпоинтер в lifespan (AsyncExitStack)
migrations/
└── env.py                    # +include_name (checkpoint* вне autogenerate)
scripts/
└── time_travel_demo.py       # NEW: демо путешествия во времени
tests/
└── test_agent_persistent.py  # NEW: HIL на sqlite in-memory
```

## Быстрый старт

```bash
uv sync                              # + langgraph-checkpoint-sqlite/postgres, psycopg
cp .env.example .env                 # LLM__OPENAI_API_KEY; в compose AGENT_CHECKPOINTER=postgres
docker compose up -d                 # app + postgres + qdrant + redis
uv run alembic upgrade head          # доменные таблицы (checkpoint* создаёт setup())
uv run uvicorn app.main:app --reload
```

Цикл «человек в цикле» через curl:

```bash
# 1) агент доходит до опасного действия и встаёт на подтверждение
curl -X POST localhost:8000/agent/chat -H 'Content-Type: application/json' \
  -d '{"message":"Вызови send_email: to=client@example.com, subject=Счёт, body=Ваш счёт","thread_id":"t1"}'
# → {"status":"interrupted","interrupt":{"type":"approve_email","preview":{...}}}

# 2) подтверждаем тем же thread_id → письмо уходит, граф завершается
curl -X POST localhost:8000/agent/resume -H 'Content-Type: application/json' \
  -d '{"thread_id":"t1","decision":true}'
```

## Проверить / тесты

```bash
uv run pytest -q
# Демо: interrupt → история чек-пойнтов → две ветки из одного входа
uv run python -m scripts.time_travel_demo
```

Студенческий README снимка m6_b4. Сервис один и тот же, растёт по чекпоинтам.
