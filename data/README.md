# Данные для индексации в Qdrant

`sample_kb.jsonl` — учебный корпус из 120 FAQ-фрагментов про техподдержку SaaS-сервиса.
Используется скриптом `scripts/load_to_qdrant.py` для демонстрации end-to-end:
загрузка → коллекция в Qdrant → поиск с фильтрами.

## Формат

Один JSON-объект на строку:

```json
{
  "source": "policy_refund.md",
  "chunk_index": 0,
  "text": "Возврат средств производится в течение 14 дней с момента оплаты...",
  "category": "billing",
  "created_at": "2026-04-01T10:00:00Z"
}
```

Поля:
- `source` — имя файла или URL источника (фильтрация по конкретному документу)
- `chunk_index` — порядок чанка внутри документа (для составления detrm. UUID id)
- `text` — сам текст чанка (возвращается в payload — на нём LLM строит ответ)
- `category` — bucket для фильтрации: `billing` / `support` / `onboarding` / `security` / `integrations` / `api`
- `created_at` — ISO 8601 timestamp для DatetimeRange-фильтров (свежесть документа)

## Замена на свои данные

На дипломе студент кладёт сюда свои документы той же структуры
(парсинг PDF/MD из своей предметки → JSONL). Поля те же:
`source`, `chunk_index`, `text`, `category`, `created_at`. По желанию —
дополнительные предметные поля (`tenant_id`, `department`, `access_level`):
их также нужно объявить в `VectorStore.payload_indexes`.
