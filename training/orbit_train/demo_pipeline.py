"""Run the trained models on real Sentinel-2 for the demo sites and export results.

For every site and year it fetches Sentinel-2 L2A (free Earth Search STAC), builds a yearly
median composite, then:
  * our Siamese U-Net (trained from scratch on OSCD)  -> change mask per year   [primary model]
  * frozen Clay + our conv head                        -> change mask per year   [second model]
  * Clay + our EuroSAT head                            -> "from -> to" land cover of the changed area
  * Clay patch features                                -> look-alike places across the district
It also exports OSCD test-set pictures (before / after / ground truth / our prediction) and the
metrics tables, so the results can be shown and checked.

    python -m orbit_train.demo_pipeline --out DRIVE/orbitwatch --sites demo_sites.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .clay import PATCH, band_indices, embed_scene, encode, load_encoder
from .data import normalise
from .models import SiameseUNet
from .train_change_clay import ConvChangeHead
from .train_land_cover import make_head

STAC_URL = "https://earth-search.aws.element84.com/v1"
# Collection 1 L2A: public HTTPS COGs (free, no AWS credentials). The L1C collection sits in a
# requester-pays bucket, so we avoid it. L2A has no cirrus band (B10): that channel is left at 0,
# which is harmless because the Siamese U-Net standardises every band per image.
COLLECTION = "sentinel-2-c1-l2a"
OSCD13 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
ASSET = {"B01": "coastal", "B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1",
         "B06": "rededge2", "B07": "rededge3", "B08": "nir", "B8A": "nir08", "B09": "nir09",
         "B10": None, "B11": "swir16", "B12": "swir22"}
YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
HA_PER_PIXEL = 0.01  # 10 m x 10 m
MAX_ITEMS = 40  # scenes per yearly composite; fewer = faster, noisier (set via --max-items)


# ----------------------------------------------------------------------------- imagery
def search_items(bbox, year):
    from pystac_client import Client

    cat = Client.open(STAC_URL)
    res = cat.search(collections=[COLLECTION], bbox=bbox, datetime=f"{year}-01-01/{year}-12-31",
                     query={"eo:cloud_cover": {"lt": 40}}, max_items=MAX_ITEMS)
    return res.item_collection()


def composite(items, bands, geobox=None, bbox=None):
    """Yearly median composite -> (len(bands), H, W) float32 on a shared grid."""
    import os

    from odc.stac import load

    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")  # public data, never sign requests
    try:
        from odc.stac import configure_rio

        configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    except Exception:  # noqa: BLE001  (older odc-stac: the env var above is enough)
        pass

    present = [(i, ASSET[b]) for i, b in enumerate(bands) if ASSET[b]]
    assets = [a for _, a in present]
    missing = [a for a in assets if a not in items[0].assets]
    if missing:
        raise RuntimeError(f"assets {missing} not in item; available: {sorted(items[0].assets)}")
    # No hand-picked CRS: odc-stac uses the scenes' native UTM zone. Later years reuse the first grid.
    kw = {"geobox": geobox} if geobox is not None else {"bbox": bbox, "resolution": 10}
    ds = load(items, bands=assets, groupby="solar_day", chunks={}, **kw)
    got = np.stack([ds[a].astype("float32").where(lambda x: x > 0).median("time").values for a in assets])
    got = np.nan_to_num(got, nan=0.0)
    if got.max() <= 3.0:  # reflectance 0..1 -> DN scale with the +1000 baseline offset (as stored)
        got = got * 10000.0 + 1000.0
    arr = np.zeros((len(bands), *got.shape[1:]), dtype=np.float32)  # unavailable bands (B10) stay 0
    for k, (i, _) in enumerate(present):
        arr[i] = got[k]
    return arr, ds.odc.geobox


# ----------------------------------------------------------------------------- models
def load_siamese(out_dir: Path, device):
    models = []
    for p in sorted(out_dir.glob("siamese_unet_seed*.pt")):
        m = SiameseUNet().to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        models.append(m.eval())
    return models


def load_conv_heads(out_dir: Path, device):
    heads = []
    for p in sorted(out_dir.glob("clay_conv_seed*.pt")):
        h = ConvChangeHead().to(device)
        h.load_state_dict(torch.load(p, map_location=device))
        heads.append(h.eval())
    return heads


@torch.no_grad()
def predict_siamese(models, before: torch.Tensor, after: torch.Tensor, device, tile=256) -> np.ndarray:
    """Tiled inference, same per-tile normalisation as evaluation. Returns prob map (H, W)."""
    _, h, w = before.shape
    prob = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            b = normalise(before[:, y:y + tile, x:x + tile])[None].to(device)
            a = normalise(after[:, y:y + tile, x:x + tile])[None].to(device)
            th, tw = b.shape[-2:]
            ph, pw = (-th) % 8, (-tw) % 8
            b, a = (F.pad(t, (0, pw, 0, ph), mode="reflect") for t in (b, a))
            p = torch.stack([torch.sigmoid(m(b, a))[0, 0, :th, :tw] for m in models]).mean(0)
            prob[y:y + th, x:x + tw] = p.cpu().numpy()
    return prob


@torch.no_grad()
def predict_clay_conv(heads, fb: torch.Tensor, fa: torch.Tensor, shape, device) -> np.ndarray:
    fb, fa = fb[None].float().to(device), fa[None].float().to(device)
    p = torch.stack([torch.sigmoid(h(fb, fa))[0, 0] for h in heads]).mean(0)
    return p[: shape[0], : shape[1]].cpu().numpy()


def save_mask(mask: np.ndarray, path: Path):
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (251, 146, 60, 170)
    Image.fromarray(rgba).save(path, optimize=True)


# ----------------------------------------------------------------------------- land cover
@torch.no_grad()
def land_cover_change(enc, lc_ckpt, before, after, mask, idx10, device):
    """Majority class of the changed area before vs after, from 64x64 chips (EuroSAT-sized)."""
    ck = torch.load(lc_ckpt, map_location="cpu")
    head = make_head(1024, len(ck["classes"]))
    head.load_state_dict(ck["state"])
    head.eval().to(device)
    mu, sd = ck["mu"].to(device), ck["sd"].to(device)

    _, h, w = before.shape
    chips = [(y, x) for y in range(0, h - 63, 64) for x in range(0, w - 63, 64)
             if mask[y:y + 64, x:x + 64].mean() > 0.4][:60]
    if not chips:
        return None, None

    def classify(img):
        batch = torch.stack([img[idx10, y:y + 64, x:x + 64] for y, x in chips])
        cls = encode(enc, batch, device)[:, 0]
        return head(((cls - mu) / sd)).argmax(1).tolist()

    top = lambda v: ck["classes"][Counter(v).most_common(1)[0][0]]  # noqa: E731
    return top(classify(before)), top(classify(after))


# ----------------------------------------------------------------------------- look-alikes
def lookalikes(enc, site, after_site_feat, mask_tokens, bbox_wide, year, device, k=5):
    from pyproj import Transformer

    items = search_items(bbox_wide, year)
    idx10 = band_indices(OSCD13)
    wide, gbox = composite(items, [OSCD13[i] for i in idx10], bbox=bbox_wide)
    feats = embed_scene(enc, torch.from_numpy(wide), device)  # (1024, h, w)
    q = after_site_feat[:, mask_tokens].float().mean(1)
    q = q / (q.norm() + 1e-6)
    f = feats.float()
    sim = (q[:, None, None] * (f / (f.norm(dim=0, keepdim=True) + 1e-6))).sum(0).numpy()

    to_ll = Transformer.from_crs(str(gbox.crs), "EPSG:4326", always_xy=True)
    out, sim_work = [], sim.copy()
    for _ in range(k):
        r, c = np.unravel_index(np.argmax(sim_work), sim_work.shape)
        px, py = (c * PATCH + PATCH / 2), (r * PATCH + PATCH / 2)
        x, y = gbox.transform * (px, py)
        lng, lat = to_ll.transform(x, y)
        d_site = np.hypot(lat - site["lat"], lng - site["lng"])
        if d_site > site["halfDeg"] * 1.5:  # skip the site itself
            out.append({"lat": float(lat), "lng": float(lng), "score": float(sim_work[r, c])})
        sim_work[max(0, r - 25):r + 25, max(0, c - 25):c + 25] = -1  # ~2 km non-max suppression
    return out


# ----------------------------------------------------------------------------- per-site run
def run_site(site, out_dir: Path, enc, siam, convs, device, base_year=YEARS[0]):
    sid = site["id"]
    d = out_dir / "demo" / sid
    d.mkdir(parents=True, exist_ok=True)
    h = site["halfDeg"]
    bbox = [site["lng"] - h, site["lat"] - h, site["lng"] + h, site["lat"] + h]
    idx10 = band_indices(OSCD13)

    comps, gbox = {}, None
    for y in YEARS:
        items = search_items(bbox, y)
        if len(items) == 0:
            raise RuntimeError(f"{sid} {y}: no Sentinel-2 scenes under 30% cloud")
        comps[y], gbox = composite(items, OSCD13, geobox=gbox, bbox=bbox)
        print(sid, y, f"{len(items)} scenes", comps[y].shape, flush=True)
    _, H, W = comps[base_year].shape

    base = torch.from_numpy(comps[base_year])
    base_feat = embed_scene(enc, base[idx10], device)
    feats = {y: embed_scene(enc, torch.from_numpy(comps[y])[idx10], device) for y in YEARS[1:]}

    models = {"siamese_unet": {"label": "Siamese U-Net (ours, from scratch)", "timeline": [], "masks": {}},
              "clay_conv": {"label": "Frozen Clay + our conv head", "timeline": [], "masks": {}}}
    final = {}
    for y in YEARS:
        for name in models:
            if y == base_year:
                models[name]["timeline"].append({"year": y, "changedHa": 0})
                continue
            after = torch.from_numpy(comps[y])
            if name == "siamese_unet":
                prob = predict_siamese(siam, base, after, device)
            else:
                prob = predict_clay_conv(convs, base_feat, feats[y], (H, W), device)
            mask = prob > 0.5
            fn = f"mask_{name}_{y}.png"
            save_mask(mask, d / fn)
            models[name]["masks"][str(y)] = fn
            models[name]["timeline"].append({"year": y, "changedHa": round(float(mask.sum() * HA_PER_PIXEL), 1)})
            if y == YEARS[-1]:
                final[name] = (mask, prob)

    for name, m in models.items():
        tl = m["timeline"]
        top_area = tl[-1]["changedHa"]
        m["confidence"] = float(final[name][1][final[name][0]].mean()) if final[name][0].any() else 0.0
        m["started"] = next((t["year"] for t in tl if top_area > 0 and t["changedHa"] >= 0.1 * top_area), None)

    mask = final["siamese_unet"][0]
    frm, to = land_cover_change(enc, out_dir / "land_cover_cls_seed0.pt", base,
                                torch.from_numpy(comps[YEARS[-1]]), mask, idx10, device)

    tok_mask = None
    if mask.any():
        th, tw = feats[YEARS[-1]].shape[1:]
        tm = F.adaptive_avg_pool2d(torch.from_numpy(mask.astype(np.float32))[None, None], (th, tw))[0, 0] > 0.3
        tok_mask = tm
    look = []
    if tok_mask is not None and tok_mask.any():
        pad = 0.15
        wide = [site["lng"] - pad, site["lat"] - pad, site["lng"] + pad, site["lat"] + pad]
        look = lookalikes(enc, site, feats[YEARS[-1]], tok_mask, wide, YEARS[-1], device)

    meta = {"id": sid, "name": site["name"], "region": site["region"], "lat": site["lat"], "lng": site["lng"],
            "halfDeg": h, "bbox": bbox, "years": YEARS, "baseYear": base_year, "primaryModel": "siamese_unet",
            "fromClass": frm, "toClass": to, "models": models, "lookalikes": look,
            "source": f"Sentinel-2 L2A (Collection 1) yearly median composites ({STAC_URL})"}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", d / "meta.json")


# ----------------------------------------------------------------------------- OSCD pictures + metrics
def export_oscd_examples(out_dir: Path, root: str, siam, device, n=6):
    """Before | after | ground truth | our prediction, on the 10 held-out OSCD test pairs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchgeo.datasets import OSCD

    from .data import split_pair
    from .metrics import change_metrics

    ds = OSCD(root=root, split="test", bands=OSCD.all_bands, download=True)
    d = out_dir / "demo" / "oscd"
    d.mkdir(parents=True, exist_ok=True)
    rgb = [OSCD.all_bands.index(b) for b in ("B04", "B03", "B02")]
    rows = []
    for i in range(min(n, len(ds))):
        s = ds[i]
        before, after = split_pair(s["image"].float())
        gt = s["mask"]
        gt = ((gt[0] if gt.ndim == 3 else gt) > 0).numpy()
        pred = predict_siamese(siam, before, after, device) > 0.5
        m = change_metrics(pred, gt)
        fig, ax = plt.subplots(1, 4, figsize=(14, 3.8))
        show = lambda t: np.clip(t[rgb].permute(1, 2, 0).numpy() / 3000, 0, 1)  # noqa: E731
        for a, im, title in zip(ax, [show(before), show(after), gt, pred],
                                ["Before", "After", "Ground truth (human labels)", f"Our model  F1 {m['f1']:.2f}"]):
            a.imshow(im, cmap="gray" if im.ndim == 2 else None)
            a.set_title(title, fontsize=10)
            a.axis("off")
        fig.tight_layout()
        fn = f"oscd_test_{i}.png"
        fig.savefig(d / fn, dpi=110)
        plt.close(fig)
        rows.append({"file": fn, **m})
    (d / "gallery.json").write_text(json.dumps(rows, indent=2))
    print("wrote OSCD gallery", len(rows))


def export_metrics(out_dir: Path):
    runs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    agg = {}
    for r in runs:
        if "test" in r:
            agg.setdefault(r["model"], []).append(r)
    change = {}
    for model, rs in agg.items():
        change[model] = {"seeds": len(rs), "params": rs[0]["params"],
                         "train_seconds": float(np.mean([r["train_seconds"] for r in rs])),
                         **{k: {"mean": float(np.mean([r["test"][k] for r in rs])),
                                "std": float(np.std([r["test"][k] for r in rs]))}
                            for k in ("f1", "iou", "precision", "recall")}}
    lc = [r for r in runs if r["model"] == "land_cover_mlp" and r["feature"] == "cls"]
    land = ({"seeds": len(lc), "params": lc[0]["params"],
             "test_acc": {"mean": float(np.mean([r["test_acc"] for r in lc])),
                          "std": float(np.std([r["test_acc"] for r in lc]))}} if lc else None)
    d = out_dir / "demo"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({"change": change, "landCover": land}, indent=2))
    print(json.dumps({"change": change, "landCover": land}, indent=2))


def main():
    global MAX_ITEMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Drive folder with trained weights")
    ap.add_argument("--sites", required=True)
    ap.add_argument("--oscd", default="/content/oscd")
    ap.add_argument("--years", type=int, nargs="+", default=YEARS, help="first year is the 'before' year")
    ap.add_argument("--max-items", type=int, default=MAX_ITEMS)
    ap.add_argument("--skip-extras", action="store_true", help="skip metrics + OSCD pictures (already exported)")
    args = ap.parse_args()
    out = Path(args.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_ITEMS = args.max_items
    YEARS[:] = sorted(args.years)

    siam = load_siamese(out, device)
    convs = load_conv_heads(out, device)
    assert siam, f"no siamese_unet_seed*.pt in {out}"
    assert convs, f"no clay_conv_seed*.pt in {out}"
    print(f"loaded {len(siam)} Siamese seeds, {len(convs)} Clay conv seeds")
    enc = load_encoder(str(out / "clay_encoder_v1.5.pt"), device)

    if not args.skip_extras:
        export_metrics(out)
        export_oscd_examples(out, args.oscd, siam, device)
    for site in json.loads(Path(args.sites).read_text()):
        run_site(site, out, enc, siam, convs, device)


if __name__ == "__main__":
    main()
