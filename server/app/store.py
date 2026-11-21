"""Read-only access to the results exported by the Colab demo pipeline."""
import io
import json
import threading
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .config import settings

HA_PER_PIXEL = 0.01  # 10 m x 10 m Sentinel-2 pixel
MIN_BLOB = 30


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def site_ids() -> list[str]:
    root = settings.data_dir
    if not root.exists():
        return []
    return sorted(p.parent.name for p in root.glob("*/meta.json"))


def load_site(site_id: str) -> dict | None:
    path = settings.data_dir / site_id / "meta.json"
    if site_id not in site_ids():  # also blocks path tricks like "../x"
        return None
    return _read_json(path)


def summary(meta: dict) -> dict:
    primary = meta["models"][meta["primaryModel"]]
    return {
        "id": meta["id"], "name": meta["name"], "region": meta["region"],
        "lat": meta["lat"], "lng": meta["lng"], "halfDeg": meta["halfDeg"],
        "fromClass": meta["fromClass"], "toClass": meta["toClass"],
        "changedHa": primary["timeline"][-1]["changedHa"],
        "confidence": primary["confidence"], "started": primary["started"],
        "lookalikes": meta["lookalikes"],
    }


def metrics() -> dict | None:
    path = settings.data_dir / "metrics.json"
    return _read_json(path) if path.exists() else None


def oscd_gallery() -> list[dict]:
    path = settings.data_dir / "oscd" / "gallery.json"
    return _read_json(path) if path.exists() else []


@lru_cache(maxsize=64)
def _prob(site_id: str, model: str, year: int) -> np.ndarray | None:
    path = settings.data_dir / site_id / f"prob_{model}_{year}.png"
    if not path.exists():
        return None
    return np.asarray(Image.open(path), dtype=np.float32) / 255.0


@lru_cache(maxsize=128)
def rethreshold(site_id: str, model: str, year: int, thr: float) -> np.ndarray | None:
    """Mask from the saved probability map at a chosen threshold (small blobs removed)."""
    prob = _prob(site_id, model, year)
    if prob is None:
        return None
    mask = prob > thr
    lab, n = ndimage.label(mask)
    if n:
        sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
        mask = np.concatenate([[False], sizes >= MIN_BLOB])[lab]
    return mask


def mask_png(mask: np.ndarray) -> bytes:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (251, 146, 60, 170)
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def retuned_timeline(meta: dict, model: str, thr: float) -> list[dict] | None:
    """Changed hectares per year at a custom threshold (None if probabilities weren't saved)."""
    out = []
    for t in meta["models"][model]["timeline"]:
        if t["year"] == meta["baseYear"]:
            out.append({"year": t["year"], "changedHa": 0})
            continue
        mask = rethreshold(meta["id"], model, t["year"], round(thr, 3))
        if mask is None:
            return None
        out.append({"year": t["year"], "changedHa": round(float(mask.sum() * HA_PER_PIXEL), 1)})
    return out


# ----------------------------------------------------------------------------- alerts
_alerts_lock = threading.Lock()
MAX_ALERTS = 200


def list_alerts() -> list[dict]:
    path = settings.alerts_file
    return _read_json(path) if path.exists() else []


def add_alert(alert: dict) -> dict:
    """Append a worker alert (newest first, capped). Returns the stored record."""
    record = {**alert, "createdAt": datetime.now(UTC).isoformat(timespec="seconds")}
    with _alerts_lock:
        alerts = [record, *list_alerts()][:MAX_ALERTS]
        settings.alerts_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = settings.alerts_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(alerts, indent=2), encoding="utf-8")
        tmp.replace(settings.alerts_file)  # atomic swap: readers never see half a file
    return record
