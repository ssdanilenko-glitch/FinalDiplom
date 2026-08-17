"""Тесты PII-маскирования для admin export."""

from app.observability.pii import mask_pii


def test_mask_email():
    assert mask_pii("write to user@example.com") == "write to [EMAIL]"


def test_mask_phone_ru_format():
    assert "[PHONE]" in mask_pii("звоните +7 (495) 123-45-67")


def test_mask_card_number():
    assert mask_pii("card 1234567812345678 ok") == "card [CARD] ok"


def test_passthrough_clean_text():
    assert mask_pii("обычный текст") == "обычный текст"


def test_empty_string():
    assert mask_pii("") == ""


def test_multiple_pii_in_one_string():
    res = mask_pii("a@b.com, +7 495 111-22-33, 1111222233334444")
    assert "[EMAIL]" in res
    assert "[PHONE]" in res
    assert "[CARD]" in res
