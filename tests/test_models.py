async def test_list_models(client):
    resp = await client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 2
    ids = {m["id"] for m in data}
    assert "gpt-5.4-mini" in ids
    for m in data:
        assert m["provider"] in {"openai", "ollama", "anthropic"}
        assert m["input_per_1m"] >= 0
        assert m["output_per_1m"] >= 0
