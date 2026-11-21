"""Activities do the I/O (HTTP); the workflow only decides what to call and in what order."""
import httpx
from temporalio import activity
from temporalio.exceptions import ApplicationError

from .config import settings

TIMEOUT = httpx.Timeout(30.0)


def make_client() -> httpx.AsyncClient:  # a function so tests can swap in a mock transport
    return httpx.AsyncClient(timeout=TIMEOUT)


def site_bbox(site: dict) -> list[float]:
    h = site["halfDeg"]
    return [site["lng"] - h, site["lat"] - h, site["lng"] + h, site["lat"] + h]


@activity.defn
async def list_sites() -> list[dict]:
    """The sites the API is watching (id, centre and footprint)."""
    async with make_client() as client:
        res = await client.get(f"{settings.api_url}/api/sites")
        res.raise_for_status()
        return [{k: s[k] for k in ("id", "name", "lat", "lng", "halfDeg")} for s in res.json()]


@activity.defn
async def find_new_scenes(site: dict, since_iso: str) -> dict:
    """Search the free Earth Search STAC for Sentinel-2 scenes over a site since a date."""
    body = {
        "collections": [settings.stac_collection],
        "bbox": site_bbox(site),
        "datetime": f"{since_iso}/..",
        "query": {"eo:cloud_cover": {"lt": settings.max_cloud_cover}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "limit": 50,
    }
    async with make_client() as client:
        res = await client.post(f"{settings.stac_url}/search", json=body)
        if res.status_code in (400, 422):  # our request is malformed: retrying won't help
            raise ApplicationError(f"STAC rejected the search: {res.text[:200]}", non_retryable=True)
        res.raise_for_status()
        items = res.json().get("features", [])

    clouds = [i["properties"].get("eo:cloud_cover") for i in items if "eo:cloud_cover" in i["properties"]]
    return {
        "siteId": site["id"],
        "siteName": site.get("name", site["id"]),
        "count": len(items),
        "latest": items[0]["properties"]["datetime"] if items else None,
        "minCloud": round(min(clouds), 1) if clouds else None,
        "since": since_iso,
    }


@activity.defn
async def post_alert(result: dict) -> None:
    """Tell the API there is fresh imagery for a site."""
    if not settings.worker_token:
        raise ApplicationError("ORBITWATCH_WORKER_TOKEN is not set", non_retryable=True)
    latest = (result["latest"] or "")[:10]
    payload = {
        "siteId": result["siteId"],
        "kind": "new_scenes",
        "newScenes": result["count"],
        "latestDate": latest or None,
        "message": (
            f"{result['count']} new Sentinel-2 scenes since {result['since'][:10]} "
            f"(latest {latest}, best cloud cover {result['minCloud']}%). Queued for a change check."
        ),
    }
    async with make_client() as client:
        res = await client.post(f"{settings.api_url}/api/alerts", json=payload,
                                headers={"X-Worker-Token": settings.worker_token})
        if res.status_code in (401, 404, 422, 503):  # config problems, not transient
            raise ApplicationError(f"API refused the alert ({res.status_code}): {res.text[:200]}",
                                   non_retryable=True)
        res.raise_for_status()


ALL_ACTIVITIES = [list_sites, find_new_scenes, post_alert]
