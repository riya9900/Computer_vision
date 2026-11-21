# OrbitWatch: Satellite Change Watchdog

CSE471 Computer Vision project, Lovely Professional University.

Watches the Earth from space and shows where land has changed (a new mine, a filling reservoir, a new city). A model we trained compares Sentinel-2 photos of the same place at two dates and marks what changed, how much, and from what to what. Clay (an open foundation model) finds look-alike places nearby. A Temporal worker checks every week for new imagery.

## What is real

| Part | Status |
|---|---|
| Siamese U-Net change detector (trained from scratch on OSCD) | real, scored below |
| Frozen Clay + our conv head / linear probe, EuroSAT land-cover head | real, scored below |
| Change masks, hectares, "from → to", look-alike places on the 4 demo sites | real model output on real Sentinel-2 (2018 vs 2023) |
| Weekly worker | real: finds new Sentinel-2 scenes per site and raises alerts. It does **not** re-run the model; that needs a GPU (see limits) |
| Site "report" button | templated text built from the numbers above, not a language model |

## Results (held-out test data, mean ± std over 3 seeds)

Change detection, OSCD test (10 pairs):

| Model | F1 | IoU | Precision | Recall | Trainable params | Train time |
|---|---|---|---|---|---|---|
| **Siamese U-Net (ours, from scratch)** | **0.544 ± 0.015** | 0.374 | 0.491 | 0.610 | 489k | ~2 min |
| Frozen Clay + conv head (ours) | 0.461 ± 0.010 | 0.299 | 0.726 | 0.338 | 1.7M | ~16 s |
| Frozen Clay + linear probe | 0.455 ± 0.001 | 0.294 | 0.477 | 0.434 | 3k | ~10 s |

Land cover, EuroSAT held-out test: **97.9 % ± 0.1** accuracy (530k-parameter head on frozen Clay features).

Train time is the measured time of each final run on a Colab GPU. Total Colab time the team spent on training (first slower attempt, 3 seeds per model, feature caching) was about 40 minutes (team-reported).

Data: OSCD (24 Sentinel-2 pairs, 13 bands; 14 train / 10 test), EuroSAT (27,000 labelled patches). Training ran on Google Colab (L4 GPU). The `/results` page in the app shows the table, plus test scenes with our prediction next to the human labels.

## Run it (3 terminals)

First, the model results must be in `server/data/demo/` (the folder `demo/` exported by `training/notebooks/colab_demo.ipynb` to `MyDrive/orbitwatch/demo`). It holds `metrics.json`, `oscd/`, and one folder per site.

```bash
# 1. API            -> http://localhost:8000/docs
cd server && uv run uvicorn app.main:app --port 8000

# 2. Web app        -> http://localhost:3000
cd client && pnpm install && pnpm dev

# 3. Weekly worker  (Temporal; needs worker/.env, see worker/.env.example)
cd worker && pnpm run dev
```

Worker extras: `pnpm run schedule` creates the weekly schedule; `pnpm run run-now` runs one check immediately and prints which sites have new scenes (their alerts then show in the web app). The API and worker must share the same `ORBITWATCH_WORKER_TOKEN` (set it in `server/.env` and `worker/.env`).

Tests: `cd server && uv run pytest` · `cd worker && uv run pytest` · `cd client && pnpm lint && npx tsc --noEmit`.

## Folders

| Folder | Purpose | Stack |
|---|---|---|
| `training/` | Models, feature caching, evaluation, Colab notebooks | PyTorch, torchgeo, Clay |
| `server/` | REST API over the results (masks, metrics, alerts) | FastAPI, uv |
| `client/` | 3D globe, before/after slider, strictness slider, results page | Next.js, react-globe.gl |
| `worker/` | Weekly Sentinel-2 scene watch | Temporal (Python SDK) |

## How the models were built

1. `training/orbit_train/train_change.py`: Siamese U-Net on OSCD (Dice + weighted BCE, crops with flips/rotations/before-after swaps).
2. `training/orbit_train/clay.py`, `cache_features.py`: run frozen Clay once on a GPU and cache features, then `train_land_cover.py` and `train_change_clay.py` train the small heads in seconds.
3. `training/orbit_train/demo_pipeline.py`: runs the models on real Sentinel-2 (L2A) for the demo sites and exports masks, hectares, land-cover change and look-alikes.

## Honest limits

- OSCD labels **urban change only**. Mines, reservoirs and salt-flat ponds are qualitative demos of the same models, not scored.
- Sentinel-2 is 10 m per pixel: only large changes are visible.
- The demo composes 2018 vs 2023 yearly composites. Season and haze differences cause false alarms, so the detection threshold for demo sites is raised to 0.75 (adjustable live with the "strictness" slider). This is a demo setting, not a tuned result.
- The models trained on L1C data (OSCD, EuroSAT); the demo runs on L2A (free public data), without the cirrus band. The Siamese U-Net standardises bands per image, so this matters little, but it is a mismatch.
- Re-running the change model on fresh scans needs Clay and a GPU (run the Colab pipeline, or a GPU instance). The worker only discovers the new scenes.
- We report that land **changed**. We do not decide whether it is illegal; a person must check.

## Deploying (AWS, optional)

The API is a plain FastAPI app and the web app a standard Next.js build. A single small EC2 instance (or two App Runner services) is enough: run the API with `uvicorn`, build the client with `NEXT_PUBLIC_API_URL` set to the API's public URL, copy `server/data/demo` onto the instance, and set `ORBITWATCH_CORS_ORIGINS` to the web app's URL. Set a budget alarm. This path has not been exercised here.


To start the project, open three terminals:

cd R:/cv/Computer_vision/server && uv run uvicorn app.main:app --port 8000
cd R:/cv/Computer_vision/client && pnpm dev
cd R:/cv/Computer_vision/worker && pnpm run dev