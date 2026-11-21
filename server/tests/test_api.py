"""API smoke tests on a tiny synthetic data folder (test fixture only, never served)."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import store
from app.config import settings
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    site = tmp_path / "demo" / "s1"
    site.mkdir(parents=True)
    prob = np.zeros((40, 40), dtype=np.uint8)
    prob[5:25, 5:25] = 230  # one confident 20x20 blob
    prob[30, 30] = 255  # a single noisy pixel that must be removed
    Image.fromarray(prob).save(site / "prob_siamese_unet_2023.png")
    Image.fromarray(np.zeros((40, 40, 4), dtype=np.uint8)).save(site / "mask_siamese_unet_2023.png")
    meta = {"id": "s1", "name": "Test", "region": "R", "lat": 0, "lng": 0, "halfDeg": 0.1,
            "baseYear": 2018, "years": [2018, 2023], "primaryModel": "siamese_unet",
            "fromClass": "Forest", "toClass": "Industrial", "lookalikes": [],
            "models": {"siamese_unet": {
                "timeline": [{"year": 2018, "changedHa": 0}, {"year": 2023, "changedHa": 4.0}],
                "masks": {"2023": "mask_siamese_unet_2023.png"}, "confidence": 0.9, "started": 2023}}}
    (site / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "demo" / "metrics.json").write_text(json.dumps({"change": {}, "landCover": None}))
    monkeypatch.setattr(settings, "data_dir", tmp_path / "demo")
    monkeypatch.setattr(settings, "alerts_file", tmp_path / "alerts.json")
    monkeypatch.setattr(settings, "worker_token", "secret")
    store._prob.cache_clear()
    store.rethreshold.cache_clear()
    return TestClient(app)


def test_health_and_sites(client):
    assert client.get("/health").json()["sites"] == 1
    sites = client.get("/api/sites").json()
    assert sites[0]["id"] == "s1" and sites[0]["changedHa"] == 4.0


def test_unknown_site_and_path_tricks(client):
    assert client.get("/api/sites/nope").status_code == 404
    assert client.get("/api/sites/..%2F..%2Fetc").status_code == 404


def test_rethreshold_removes_noise_and_counts_area(client):
    # 20x20 blob = 400 px = 4 ha; the lone pixel is dropped
    site = client.get("/api/sites/s1?thr=0.5").json()
    assert site["models"]["siamese_unet"]["timeline"][-1]["changedHa"] == 4.0
    r = client.get("/api/sites/s1/mask/siamese_unet/2023.png?thr=0.5")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get("/api/sites/s1/mask/siamese_unet/2023.png").status_code == 200
    assert client.get("/api/sites/s1/mask/siamese_unet/1999.png").status_code == 404


def test_metrics(client):
    assert client.get("/api/metrics").status_code == 200
    assert client.get("/api/oscd").json() == []


ALERT = {"siteId": "s1", "newScenes": 3, "latestDate": "2026-11-10", "message": "3 new scenes"}


def test_alerts_need_the_worker_token(client, monkeypatch):
    assert client.get("/api/alerts").json() == []
    assert client.post("/api/alerts", json=ALERT).status_code == 401
    assert client.post("/api/alerts", json=ALERT, headers={"X-Worker-Token": "wrong"}).status_code == 401
    ok = client.post("/api/alerts", json=ALERT, headers={"X-Worker-Token": "secret"})
    assert ok.status_code == 201 and "createdAt" in ok.json()
    assert client.get("/api/alerts?site_id=s1").json()[0]["newScenes"] == 3
    assert client.get("/api/alerts?site_id=other").json() == []
    bad = client.post("/api/alerts", json={**ALERT, "siteId": "nope"}, headers={"X-Worker-Token": "secret"})
    assert bad.status_code == 404
    monkeypatch.setattr(settings, "worker_token", None)
    assert client.post("/api/alerts", json=ALERT, headers={"X-Worker-Token": "secret"}).status_code == 503
