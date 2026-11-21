import json
import uuid

import httpx
import pytest
from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment, WorkflowEnvironment
from temporalio.worker import Worker

from orbitwatch_worker import activities
from orbitwatch_worker.config import settings
from orbitwatch_worker.workflows import WeeklyWatchWorkflow

SITE = {"id": "s1", "name": "Site One", "lat": 10.0, "lng": 20.0, "halfDeg": 0.1}


def mock_http(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(activities, "make_client", lambda: httpx.AsyncClient(transport=transport))


async def test_find_new_scenes_summarises_the_stac_answer(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        feats = [{"properties": {"datetime": "2026-11-10T02:30:00Z", "eo:cloud_cover": 12.4}},
                 {"properties": {"datetime": "2026-11-05T02:30:00Z", "eo:cloud_cover": 3.1}}]
        return httpx.Response(200, json={"features": feats})

    mock_http(monkeypatch, handler)
    out = await ActivityEnvironment().run(activities.find_new_scenes, SITE, "2026-11-01T00:00:00Z")
    assert out["count"] == 2 and out["latest"].startswith("2026-11-10") and out["minCloud"] == 3.1
    assert seen["body"]["bbox"] == [19.9, 9.9, 20.1, 10.1]
    assert seen["body"]["datetime"] == "2026-11-01T00:00:00Z/.."


async def test_find_new_scenes_with_no_results(monkeypatch):
    mock_http(monkeypatch, lambda r: httpx.Response(200, json={"features": []}))
    out = await ActivityEnvironment().run(activities.find_new_scenes, SITE, "2026-11-01T00:00:00Z")
    assert out["count"] == 0 and out["latest"] is None and out["minCloud"] is None


async def test_post_alert_sends_the_token_and_does_not_retry_config_errors(monkeypatch):
    sent = {}

    def ok(request: httpx.Request) -> httpx.Response:
        sent["token"] = request.headers["x-worker-token"]
        sent["body"] = json.loads(request.content)
        return httpx.Response(201, json={})

    result = {"siteId": "s1", "count": 3, "latest": "2026-11-10T02:30:00Z", "minCloud": 3.1,
              "since": "2026-11-03T00:00:00Z"}
    monkeypatch.setattr(settings, "worker_token", "secret")
    mock_http(monkeypatch, ok)
    await ActivityEnvironment().run(activities.post_alert, result)
    assert sent["token"] == "secret" and sent["body"]["newScenes"] == 3 and sent["body"]["latestDate"] == "2026-11-10"

    mock_http(monkeypatch, lambda r: httpx.Response(401, json={"detail": "bad worker token"}))
    with pytest.raises(ApplicationError) as err:
        await ActivityEnvironment().run(activities.post_alert, result)
    assert err.value.non_retryable

    monkeypatch.setattr(settings, "worker_token", None)
    with pytest.raises(ApplicationError):
        await ActivityEnvironment().run(activities.post_alert, result)


async def test_workflow_alerts_only_sites_with_new_scenes():
    alerts = []

    @activity.defn(name="list_sites")
    async def fake_list_sites() -> list[dict]:
        return [SITE, {**SITE, "id": "s2"}]

    @activity.defn(name="find_new_scenes")
    async def fake_find(site: dict, since_iso: str) -> dict:
        return {"siteId": site["id"], "count": 2 if site["id"] == "s1" else 0, "latest": None,
                "minCloud": None, "since": since_iso}

    @activity.defn(name="post_alert")
    async def fake_alert(result: dict) -> None:
        alerts.append(result["siteId"])

    env = await WorkflowEnvironment.start_time_skipping()
    async with env, Worker(env.client, task_queue="t", workflows=[WeeklyWatchWorkflow],
                           activities=[fake_list_sites, fake_find, fake_alert]):
        out = await env.client.execute_workflow(WeeklyWatchWorkflow.run, 7,
                                                id=f"wf-{uuid.uuid4()}", task_queue="t")
    assert out["sitesChecked"] == 2 and out["sitesWithNewScenes"] == ["s1"]
    assert alerts == ["s1"]
