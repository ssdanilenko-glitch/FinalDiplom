"""Юнит-тесты чистых функций офлайн-контура индексации (без Qdrant/OpenAI)."""

from llama_index.core.schema import Document

from app.services.ingestion import (
    EXCLUDED_EMBED_KEYS,
    clean,
    department_from_path,
    doc_type_from_path,
    enrich,
    file_metadata,
    version_from_filename,
)


def test_clean_strips_footer_and_joins_hyphenation() -> None:
    raw = "Регламент возврата Стр. 12 из 47 авто-\nмобиль доступен https://x.io/a тут"
    out = clean(raw)
    assert "Стр. 12 из 47" not in out
    assert "автомобиль" in out
    assert "https://" not in out


def test_clean_collapses_blank_lines() -> None:
    assert clean("a\n\n\n\n\nb") == "a\n\nb"


def test_department_from_path_uses_top_folder() -> None:
    assert department_from_path("data/finance/2025/policy.pdf") == "finance"
    assert department_from_path("knowledge_base/support/faq.md") == "support"


def test_department_from_path_defaults_to_general() -> None:
    assert department_from_path("/tmp/loose_file.pdf") == "general"
    assert department_from_path("data") == "general"


def test_doc_type_from_path() -> None:
    assert doc_type_from_path("a/b/policy.PDF") == "pdf"
    assert doc_type_from_path("note.md") == "md"
    assert doc_type_from_path("no_ext") == "unknown"


def test_version_from_filename() -> None:
    assert version_from_filename("policy_2025_v3.pdf") == "2025_v3"
    assert version_from_filename("plain_doc.pdf") == "unversioned"


def test_file_metadata_has_filter_fields() -> None:
    meta = file_metadata("data/hr/2025/onboarding_2025_v2.docx")
    assert meta["department"] == "hr"
    assert meta["doc_type"] == "docx"
    assert meta["version"] == "2025_v2"
    assert meta["visibility"] == "internal"
    assert meta["source"] == "onboarding_2025_v2.docx"


def test_enrich_cleans_text_and_excludes_technical_keys() -> None:
    docs = [Document(text="Тариф Стр. 3 из 9 описан тут", metadata={"department": "billing"})]
    out = enrich(docs)
    assert "Стр. 3 из 9" not in out[0].text
    assert out[0].excluded_embed_metadata_keys == EXCLUDED_EMBED_KEYS
    assert out[0].excluded_llm_metadata_keys == EXCLUDED_EMBED_KEYS
