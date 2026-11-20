from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import store
from .config import settings

app = FastAPI(title="OrbitWatch API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "sites": len(store.site_ids()), "data_dir": str(settings.data_dir)}


@app.get("/api/sites")
def list_sites() -> list[dict]:
    return [store.summary(store.load_site(i)) for i in store.site_ids()]


@app.get("/api/sites/{site_id}")
def get_site(site_id: str, thr: float | None = Query(None, ge=0.05, le=0.99)) -> dict:
    meta = store.load_site(site_id)
    if meta is None:
        raise HTTPException(404, "unknown site")
    if thr is not None:  # re-threshold the primary model from its saved probabilities
        model = meta["primaryModel"]
        timeline = store.retuned_timeline(meta, model, thr)
        if timeline is not None:
            meta["models"][model]["timeline"] = timeline
    return meta


@app.get("/api/sites/{site_id}/mask/{model}/{year}.png")
def get_mask(site_id: str, model: str, year: int, thr: float | None = Query(None, ge=0.05, le=0.99)) -> Response:
    meta = store.load_site(site_id)
    if meta is None or model not in meta["models"]:
        raise HTTPException(404, "unknown site or model")
    if thr is not None:
        mask = store.rethreshold(site_id, model, year, round(thr, 3))
        if mask is not None:
            return Response(store.mask_png(mask), media_type="image/png",
                            headers={"Cache-Control": "public, max-age=300"})
    fn = meta["models"][model]["masks"].get(str(year))
    if fn is None:
        raise HTTPException(404, "no mask for that year")
    return FileResponse(settings.data_dir / site_id / fn, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=300"})


@app.get("/api/metrics")
def get_metrics() -> dict:
    data = store.metrics()
    if data is None:
        raise HTTPException(404, "metrics.json not found; run the Colab demo pipeline")
    return data


@app.get("/api/oscd")
def get_oscd() -> list[dict]:
    return store.oscd_gallery()


@app.get("/api/oscd/{filename}")
def get_oscd_image(filename: str) -> FileResponse:
    if filename not in {r["file"] for r in store.oscd_gallery()}:
        raise HTTPException(404, "unknown image")
    return FileResponse(settings.data_dir / "oscd" / filename, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=3600"})
