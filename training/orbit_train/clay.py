"""Clay v1.5 encoder helpers (frozen feature extractor).

API verified against https://github.com/Clay-foundation/model (claymodel/model.py,
claymodel/finetune/segment/factory.py). Wavelengths are in micrometres, as in the
repo's own fine-tuning code.

CLI (Colab):
    python -m orbit_train.clay extract --out /content/drive/MyDrive/orbitwatch/clay_encoder_v1.5.pt
"""
import argparse
import re

import torch
import torch.nn.functional as F

# Sentinel-2 bands Clay was trained on (it has no B01/B09/B10), in Clay's order.
BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
WAVES_UM = (0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19)
MEAN = (1105.0, 1355.0, 1552.0, 1887.0, 2422.0, 2630.0, 2743.0, 2785.0, 2388.0, 1835.0)
STD = (1809.0, 1757.0, 1888.0, 1870.0, 1732.0, 1697.0, 1742.0, 1648.0, 1470.0, 1379.0)
GSD = 10.0
PATCH = 8
DIM = 1024

HF_REPO, HF_FILE = "made-with-clay/Clay", "v1.5/clay-v1.5.ckpt"


def band_indices(all_band_names) -> list[int]:
    """Indices of Clay's 10 bands inside a dataset's band list."""
    names = list(all_band_names)
    return [names.index(b) for b in BANDS]


def build_encoder():
    from claymodel.model import Encoder

    return Encoder(mask_ratio=0.0, patch_size=PATCH, shuffle=False, dim=DIM,
                   depth=24, heads=16, dim_head=64, mlp_ratio=4.0)


def extract_encoder(out_path: str) -> None:
    """Download the full Lightning checkpoint, keep only encoder weights (~1.2 GB)."""
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(HF_REPO, HF_FILE)
    # Official Clay checkpoint (Lightning format, needs weights_only=False to unpickle).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {re.sub(r"^model\.encoder\.", "", k): v
          for k, v in ckpt["state_dict"].items() if k.startswith("model.encoder.")}
    enc = build_encoder()
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"encoder keys missing from checkpoint: {missing[:5]}...")
    print(f"encoder ok, unexpected keys: {len(unexpected)}")
    torch.save(sd, out_path)
    print("saved", out_path)


def load_encoder(path: str, device: str):
    enc = build_encoder()
    enc.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    enc.eval().requires_grad_(False)
    return enc.to(device)


def _autocast_dtype():
    if not torch.cuda.is_available():
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@torch.no_grad()
def encode(enc, pixels: torch.Tensor, device: str) -> torch.Tensor:
    """pixels: raw reflectance-scale (B, 10, H, W), H and W multiples of 8.

    Returns tokens (B, 1 + H/8 * W/8, 1024): class token first, then patch tokens.
    """
    b = pixels.shape[0]
    mean = torch.tensor(MEAN).view(1, -1, 1, 1)
    std = torch.tensor(STD).view(1, -1, 1, 1)
    cube = {
        "pixels": ((pixels.float() - mean) / std).to(device),
        "time": torch.zeros(b, 4),    # unknown acquisition time/place: zeros are allowed
        "latlon": torch.zeros(b, 4),
        "gsd": torch.tensor(GSD),
        "waves": torch.tensor(WAVES_UM),
    }
    dtype = _autocast_dtype()
    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
        tokens = enc(cube)[0]
    return tokens.float()


@torch.no_grad()
def embed_scene(enc, img: torch.Tensor, device: str, tile: int = 256, batch: int = 8) -> torch.Tensor:
    """Full scene (10, H, W) -> patch-token map (1024, ceil(H/8), ceil(W/8)), fp16 on CPU.

    The scene is edge-padded to a multiple of `tile` and encoded tile by tile.
    """
    _, h, w = img.shape
    ph, pw = (-h) % tile, (-w) % tile
    x = F.pad(img[None], (0, pw, 0, ph), mode="replicate")[0]
    gh, gw = x.shape[1] // tile, x.shape[2] // tile
    g = tile // PATCH
    tiles = [x[:, i * tile:(i + 1) * tile, j * tile:(j + 1) * tile]
             for i in range(gh) for j in range(gw)]
    out = torch.zeros(DIM, gh * g, gw * g, dtype=torch.float16)
    for s in range(0, len(tiles), batch):
        chunk = torch.stack(tiles[s:s + batch])
        tok = encode(enc, chunk, device)[:, 1:]                      # drop class token
        tok = tok.reshape(len(chunk), g, g, DIM).permute(0, 3, 1, 2)  # (B, D, g, g)
        for k in range(len(chunk)):
            i, j = divmod(s + k, gw)
            out[:, i * g:(i + 1) * g, j * g:(j + 1) * g] = tok[k].half().cpu()
    return out[:, : -(-h // PATCH), : -(-w // PATCH)].contiguous()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract"])
    ap.add_argument("--out", required=True)
    extract_encoder(ap.parse_args().out)
