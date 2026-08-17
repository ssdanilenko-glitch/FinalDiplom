"""Маскирование PII (email / phone / card) в строках.

Учебная заглушка вместо полноценного Presidio. Используется при экспорте
истории через admin /export, чтобы не утекали персональные данные в выгрузку.
"""

import re

# ВАЖНО: порядок имеет значение. Card (16 digits подряд) ловим первым,
# иначе телефонный regex выгрызет первые ~10 цифр и оставит хвост вида
# "[PHONE]678". Email тоже впереди — он надёжно отличим от чисел.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w.\-]+@[\w.\-]+\.\w+"), "[EMAIL]"),
    (re.compile(r"\b\d{16}\b"), "[CARD]"),
    (
        re.compile(
            r"\+?\d{1,3}[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
        ),
        "[PHONE]",
    ),
]


def mask_pii(text: str) -> str:
    if not text:
        return text
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text
