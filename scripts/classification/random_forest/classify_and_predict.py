import os
import re
import numpy as np
import rasterio
from rasterio.enums import Resampling
from sklearn.ensemble import RandomForestClassifier
import joblib

# --------------------------
# Paths
# --------------------------
DOWNLOADS = r"C:\Users\kiran\Downloads"
BOUND_FP  = os.path.join(DOWNLOADS, "BENGALURU (1).geojson")

MASK_DIR  = os.path.join(DOWNLOADS, "test_masks")     # masks you already generated
OUT_DIR   = os.path.join(DOWNLOADS, "rf_predicted_masks")   # output folder
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# Utility functions
# --------------------------
def find_tifs(year):
    """Return dict containing all tif paths for a given year."""
    files = os.listdir(DOWNLOADS)
    out = {}
    for f in files:
        m = re.search(rf"Landsat_{year}_(RGBNIR|NDVI|NDWI|NDBI)\.tif", f)
        if m:
            out[m.group(1)] = os.path.join(DOWNLOADS, f)
    return out


def read_band(fp):
    with rasterio.open(fp) as src:
        arr = src.read(1).astype("float32")
    return np.nan_to_num(arr, nan=-1)


def stack_features(feats_dict):
    """Return (H,W,B) stack of features."""
    arrs = [read_band(feats_dict[k]) for k in ["RGBNIR", "NDVI", "NDWI", "NDBI"]]
    return np.stack(arrs, axis=-1)


# --------------------------
# TRAINING YEARS: 2014–2024
# --------------------------
TRAIN_YEARS = list(range(2014, 2025))

print("\n🚀 Collecting training data from 2014–2024...\n")

X_list = []
y_list = []

for year in TRAIN_YEARS:
    print(f"\n→ YEAR {year}")

    tif_map = find_tifs(year)
    if len(tif_map) != 4:
        print("  ❌ Missing spectral images → skipping")
        continue

    mask_fp = os.path.join(MASK_DIR, f"{year}_mask.tif")
    if not os.path.exists(mask_fp):
        print("  ❌ Mask missing → skipping")
        continue

    # Load features
    feats = stack_features(tif_map)
    H, W, B = feats.shape
    feats2d = feats.reshape(-1, B)

    # Load mask
    with rasterio.open(mask_fp) as src:
        mask = src.read(1).reshape(-1)

    # Remove outside (0)
    valid = mask > 0
    X_list.append(feats2d[valid])
    y_list.append(mask[valid])

    print(f"  ✔ Loaded training pixels: {valid.sum():,}")

# Combine training data
X_train = np.concatenate(X_list, axis=0)
y_train = np.concatenate(y_list, axis=0)

print("\n📊 Final training dataset size:")
print("  X:", X_train.shape)
print("  y:", y_train.shape)


# --------------------------
# TRAIN RANDOM FOREST
# --------------------------
print("\n🌲 Training RandomForest model...\n")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

rf.fit(X_train, y_train)

model_fp = os.path.join(OUT_DIR, "rf_bangalore_model.pkl")
joblib.dump(rf, model_fp)

print("\n✅ Model saved at:", model_fp)


# --------------------------
# PREDICT 1988–2013
# --------------------------
PRED_YEARS = list(range(1988, 2014))

print("\n🚀 Starting prediction for 1988–2013...\n")

for year in PRED_YEARS:
    print(f"\n→ Predicting YEAR {year}")

    tif_map = find_tifs(year)
    if len(tif_map) != 4:
        print("  ❌ Missing TIFs → skipping")
        continue

    feats = stack_features(tif_map)
    H, W, B = feats.shape
    feats2d = feats.reshape(-1, B)

    print("  ✔ Running prediction...")
    pred = rf.predict(feats2d)

    # Reshape to mask
    pred_img = pred.reshape(H, W).astype("uint8")

    # Save
    out_fp = os.path.join(OUT_DIR, f"{year}_mask.tif")
    with rasterio.open(tif_map["RGBNIR"]) as src:
        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": "uint8"})

        with rasterio.open(out_fp, "w", **meta) as dst:
            dst.write(pred_img, 1)

    print("  ✔ Saved:", out_fp)

print("\n🎉 ALL DONE! 1988–2013 masks generated using RF model.\n")
