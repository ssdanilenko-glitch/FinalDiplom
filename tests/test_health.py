async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_ready_when_redis_up(client):
    resp = await client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "redis": "up"}


async def test_ready_when_redis_down(client, mock_cache):
    mock_cache.ping.side_effect = ConnectionError("redis down")
    resp = await client.get("/ready")
    assert resp.status_code == 503
    assert resp.json() == {"status": "degraded", "redis": "down"}
