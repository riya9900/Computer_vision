# training

Models for OrbitWatch, trained on Google Colab (free T4); weights and metrics saved to Drive.

1. `orbit_train/train_change.py` - Siamese U-Net trained from scratch on OSCD (our own model).
2. (next) Clay feature caching, land-cover head on EuroSAT, change head on Clay features.
3. (next) baselines (spectral differencing) and a results table across seeds.

Run on Colab: open `notebooks/colab_train.ipynb`.
