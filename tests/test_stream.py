from unittest.mock import MagicMock


async def test_chat_stream(client, mock_llm):
    async def fake_stream():
        for text in ["При", "вет", "!"]:
            yield MagicMock(
                choices=[MagicMock(delta=MagicMock(content=text))],
                usage=None,
            )
        # Финальный кадр с usage (stream_options.include_usage=True)
        yield MagicMock(
            choices=[],
            usage=MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )

    mock_llm.chat.completions.create.return_value = fake_stream()

    async with client.stream(
        "POST",
        "/chat/stream",
        json={"messages": [{"role": "user", "content": "hi"}]},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        chunks = []
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                chunks.append(line[len("data: ") :])

    text_chunks = [c for c in chunks if c != "[DONE]"]
    assert any("При" in c for c in text_chunks)
    assert any("usage" in c for c in text_chunks)
    assert chunks[-1] == "[DONE]"


async def test_chat_stream_content_type_headers(client, mock_llm):
    async def fake_stream():
        yield MagicMock(
            choices=[MagicMock(delta=MagicMock(content="hi"))],
            usage=None,
        )

    mock_llm.chat.completions.create.return_value = fake_stream()

    async with client.stream(
        "POST",
        "/chat/stream",
        json={"messages": [{"role": "user", "content": "hi"}]},
    ) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers.get("x-accel-buffering") == "no"
        assert resp.headers.get("cache-control") == "no-cache"
        # Дочитываем тело, чтобы корректно закрыть стрим.
        async for _ in resp.aiter_lines():
            pass
