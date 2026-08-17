"""Генератор учебного корпуса из 120 FAQ-фрагментов про SaaS-техподдержку.

Запуск:
    python data/generate_sample.py

Создаёт `data/sample_kb.jsonl` с 120 точками — реалистичный объём
для проверки скрипта загрузки (homework требует 100+). На дипломе
заменяется на парсинг своих PDF/MD/HTML.
"""


import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

random.seed(42)

OUT = Path(__file__).resolve().parent / "sample_kb.jsonl"


SOURCES: list[tuple[str, str, str, str]] = [
    ("policy_refund.md", "billing", "2026-01-15", "Политика возвратов"),
    ("policy_pricing.md", "billing", "2026-03-10", "Тарифы и оплата"),
    ("policy_security.md", "security", "2025-11-20", "Политика безопасности"),
    ("policy_gdpr.md", "security", "2025-09-01", "Соответствие GDPR"),
    ("api_quickstart.md", "api", "2026-02-05", "API: быстрый старт"),
    ("api_authentication.md", "api", "2026-02-15", "Аутентификация API"),
    ("api_rate_limits.md", "api", "2026-04-12", "Rate limits API"),
    ("api_webhooks.md", "api", "2026-03-22", "Webhooks: настройка"),
    ("integrations_slack.md", "integrations", "2025-12-05", "Интеграция со Slack"),
    ("integrations_github.md", "integrations", "2026-01-30", "Интеграция с GitHub"),
    ("integrations_jira.md", "integrations", "2026-04-01", "Интеграция с Jira"),
    ("onboarding_workspace.md", "onboarding", "2026-02-20", "Создание workspace"),
    ("onboarding_invite_team.md", "onboarding", "2026-03-05", "Приглашение команды"),
    ("onboarding_roles.md", "onboarding", "2026-03-15", "Роли пользователей"),
    ("support_login_issues.md", "support", "2026-04-25", "Проблемы с входом"),
    ("support_password_reset.md", "support", "2026-04-20", "Сброс пароля"),
    ("support_2fa.md", "support", "2026-04-10", "Двухфакторная аутентификация"),
    ("support_billing_dispute.md", "billing", "2026-04-18", "Спор по списанию"),
    ("support_data_export.md", "security", "2026-03-28", "Экспорт данных"),
    ("support_account_deletion.md", "security", "2026-04-05", "Удаление аккаунта"),
]


CHUNK_TEMPLATES: list[str] = [
    "{title}. Краткое описание раздела. Эта часть документа объясняет основные принципы и сценарии использования.",
    "{title}: пошаговая инструкция. Откройте настройки → найдите соответствующий раздел → выполните указанные действия.",
    "{title} — частые вопросы. Большинство пользователей сталкивается с задачами, описанными в этом разделе.",
    "{title}: ограничения и нюансы. Учитывайте, что некоторые операции требуют прав администратора workspace.",
    "{title} в enterprise-плане. Расширенные возможности доступны на тарифах Business и Enterprise.",
    "{title}: troubleshooting. Если что-то не работает, проверьте логи и обратитесь в поддержку через intercom.",
]


EXTRAS: dict[str, list[str]] = {
    "billing": [
        " Возврат средств производится в течение 14 дней с момента оплаты, при условии что услуга не была использована более чем на 20%.",
        " Тарификация идёт по числу активных пользователей в workspace на 1-е число месяца.",
        " Оплата принимается картами Visa/Mastercard/Mir, через банковский перевод для enterprise-планов.",
        " Pro-rate расчёт применяется при добавлении новых пользователей в середине биллингового цикла.",
        " Free tier ограничен 3 активными пользователями и 100 МБ хранилища.",
        " Apple Pay и Google Pay доступны в мобильном приложении.",
    ],
    "security": [
        " Все данные шифруются в покое (AES-256) и при передаче (TLS 1.3).",
        " Хранение данных в РФ для российских клиентов; ЕС-резидентность для GDPR.",
        " Аудит-логи доступны administrator'ам в течение 90 дней.",
        " Удаление аккаунта запускает 30-дневный grace period перед безвозвратной очисткой.",
        " SSO через SAML 2.0 и SCIM 2.0 доступны на Enterprise-плане.",
        " Bug bounty программа открыта — отчёты принимаются на security@example.com.",
    ],
    "api": [
        " Базовый URL API: https://api.example.com/v2. Все эндпоинты используют JSON.",
        " API-ключ передаётся в заголовке Authorization: Bearer <KEY>.",
        " Rate limit по умолчанию — 60 запросов в минуту для Free, 600 — для Pro, 6000 — для Enterprise.",
        " Webhooks подписываются HMAC-SHA256 и доставляются с retry exponential backoff.",
        " Pagination через параметры cursor и limit; max limit = 100.",
        " Async-операции возвращают job_id, статус читается через GET /jobs/{id}.",
    ],
    "integrations": [
        " Установка через OAuth 2.0: подтверждение прав занимает 30 секунд.",
        " Уведомления отправляются в выбранный канал; формат настраивается шаблонами.",
        " Двусторонняя синхронизация задач: статус, исполнитель, теги.",
        " Изменения видны в обе стороны в течение 5 секунд через webhooks.",
        " Поддерживаемые версии: Slack >= 2024-Q2, GitHub Enterprise 3.10+, Jira Cloud.",
        " Для on-premise GitHub Enterprise — отдельная сетевая настройка proxy.",
    ],
    "onboarding": [
        " Создание workspace занимает менее минуты; доменное имя выбирается при регистрации.",
        " Приглашения отправляются по email; ссылка активна 7 дней.",
        " Роли по умолчанию: Owner, Admin, Member, Viewer. Кастомные роли — на Enterprise.",
        " Workspace owner не удаляется без передачи прав другому admin.",
        " Tutorial для новых членов: 5-минутный walkthrough при первом входе.",
        " Шаблоны workspace ускоряют запуск типовых сценариев: support, sales, dev.",
    ],
    "support": [
        " Восстановление пароля занимает до 5 минут; письмо отправляется на основной email.",
        " Если не приходит письмо — проверьте папку spam и whitelisting нашего домена.",
        " 2FA через TOTP-приложение (Google Authenticator, Authy) или SMS.",
        " Backup-коды генерируются один раз при включении 2FA — сохраните их.",
        " Если потеряли все способы 2FA — обратитесь в поддержку с подтверждением личности.",
        " Время реакции поддержки: Free до 48 часов, Pro до 8 часов, Enterprise до 1 часа.",
    ],
}


def main() -> None:
    points: list[dict] = []
    for source, category, base_date, title in SOURCES:
        base_dt = datetime.fromisoformat(base_date).replace(tzinfo=timezone.utc)
        for chunk_index, tmpl in enumerate(CHUNK_TEMPLATES):
            extra = random.choice(EXTRAS[category])
            text = tmpl.format(title=title) + extra
            created_at = base_dt + timedelta(days=chunk_index)
            points.append(
                {
                    "source": source,
                    "chunk_index": chunk_index,
                    "text": text,
                    "category": category,
                    "created_at": created_at.isoformat().replace("+00:00", "Z"),
                }
            )

    with OUT.open("w", encoding="utf-8") as f:
        for p in points:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Записано {len(points)} точек в {OUT}")


if __name__ == "__main__":
    main()
