# Vector store — отчёт

Шаблон. Заполняется после выполнения скриптов из проекта. Цифры и id —
из реального прогона на своей предметке (не на учебном корпусе).

## Конфигурация

- **Движок:** Qdrant `qdrant/qdrant:v1.14.0`
- **Embedding-модель:** `text-embedding-3-small` (dim=1536)
- **Метрика:** cosine
- **Коллекция:** `documents`
- **HNSW:** `m=16`, `ef_construct=100` — defaults Qdrant (см. обоснование ниже)
- **Размер корпуса:** _подставить из `client.get_collection().points_count`_

## Метрика: cosine vs dot product

Скрипт `scripts/compare_metrics.py` создаёт временные коллекции
`documents_cosine` и `documents_dot` на одних и тех же векторах,
прогоняет 5 запросов и собирает top-5.

| Запрос | top-5 cosine | top-5 dot | Совпало |
|--------|--------------|-----------|---------|
| Как вернуть деньги за подписку | `id1, id2, ...` | `id1, id2, ...` | ✓ / ✗ |
| Сбросить пароль если не приходит письмо | ... | ... | ✓ / ✗ |
| Какой rate limit на API для бесплатного плана | ... | ... | ✓ / ✗ |
| Где найти аудит-логи действий пользователей | ... | ... | ✓ / ✗ |
| Как подключить интеграцию со Slack | ... | ... | ✓ / ✗ |

**Что осталось в production:** _COSINE / DOT_

**Почему:** _Embeddings OpenAI нормализованы (||v||=1) — cosine и dot дают
идентичное ранжирование. Оставляю COSINE: явный контракт для модели
учили на cosine similarity, читаемо в коде, дефолт в большинстве SDK._

## Примеры фильтров

### 1. Match по строке: `category = "billing"`

```python
from qdrant_client.models import FieldCondition, Filter, MatchValue

flt = Filter(
    must=[FieldCondition(key="category", match=MatchValue(value="billing"))]
)
hits = await store.search(query_vector=qv, top_k=3, query_filter=flt)
```

**Топ-3 для запроса «Как вернуть деньги»:**
1. `id...` — `policy_refund.md` chunk 0 — _фрагмент текста_
2. `id...` — `policy_refund.md` chunk 1 — _фрагмент_
3. `id...` — `support_billing_dispute.md` chunk 0 — _фрагмент_

### 2. Range по дате: свежее 30 дней

```python
from datetime import datetime, timedelta, timezone
from qdrant_client.models import DatetimeRange, FieldCondition, Filter

cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
flt = Filter(
    must=[FieldCondition(key="created_at", range=DatetimeRange(gte=cutoff))]
)
hits = await store.search(query_vector=qv, top_k=3, query_filter=flt)
```

**Что меняется:** _без фильтра в топе политика 2025-09 (устарела). С
фильтром тот же запрос возвращает свежие документы апреля-мая 2026._

### 3. Композитный must + must_not: только нужный тенант, без архива

```python
from qdrant_client.models import FieldCondition, Filter, MatchValue

flt = Filter(
    must=[FieldCondition(key="tenant_id", match=MatchValue(value="acme-corp"))],
    must_not=[FieldCondition(key="status", match=MatchValue(value="archived"))],
)
hits = await store.search(query_vector=qv, top_k=3, query_filter=flt)
```

**Топ-3 для запроса «политика возврата»:** _id..., id..., id..._

## HNSW: обоснование параметров

Параметры оставлены defaults Qdrant: `m=16`, `ef_construct=100`. Обоснование:

- На корпусе **_N_** точек (10k–100k) recall с defaults — **_X%_** (замерено
  на golden set из 20 запросов с известными ground-truth top-5).
- Latency p95 — **_Y мс_** на запрос (через `client.query_points` без
  фильтров, без квантизации).
- Поднимать `m` или `ef_construct` нет смысла: запас по recall и latency
  достаточный для целевого пользовательского сценария.

При росте корпуса до 1M+ — добавить scalar quantization
(`quantization_config=ScalarQuantization(...)`) и поднять `ef_search`
до 100–128. Сейчас — overkill.

## pgvector как альтернатива (опционально)

_Заполнить, если выполнено задание 7. Latency на тех же 5 запросах:_

| Запрос | Qdrant p50 (мс) | pgvector p50 (мс) |
|--------|-----------------|-------------------|
| ... | ... | ... |

**Финальный выбор и почему:** _оставляю Qdrant / переезжаю на pgvector — обоснование._
