import os
import rasterio
import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from tqdm import tqdm
from rasterio import features

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT = r"C:\Users\kiran\Downloads"
MASK_DIR = ROOT
OUT_DIR = os.path.join(ROOT, "final_classified_masks", "predictions")
os.makedirs(OUT_DIR, exist_ok=True)

BOUNDARY_FP = os.path.join(ROOT, "BENGALURU (1).geojson")

TRAIN_YEARS = list(range(2014, 2025))
PREDICT_YEARS = list(range(1988, 2014))

# ─────────────────────────────────────────────
# LOAD BOUNDARY
# ─────────────────────────────────────────────
boundary = gpd.read_file(BOUNDARY_FP)

# ─────────────────────────────────────────────
# FEATURE STACK FUNCTION
# ─────────────────────────────────────────────
def load_features(year):
    files = {
        "rgbnir": f"Landsat_{year}_RGBNIR.tif",
        "ndvi":   f"Landsat_{year}_NDVI.tif",
        "ndwi":   f"Landsat_{year}_NDWI.tif",
        "ndbi":   f"Landsat_{year}_NDBI.tif",
    }

    arrays = []
    for f in files.values():
        with rasterio.open(os.path.join(ROOT, f)) as src:
            arrays.append(np.nan_to_num(src.read(), nan=0))

    with rasterio.open(os.path.join(ROOT, files["rgbnir"])) as src:
        H, W = src.height, src.width
        transform = src.transform
        crs = src.crs

    # Spatial features
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    arrays.append(rows[None, :, :])
    arrays.append(cols[None, :, :])

    # Rasterize boundary → inside = 1, outside = 0
    boundary_pr = boundary.to_crs(crs)
    inside_mask = features.rasterize(
        [(g, 1) for g in boundary_pr.geometry],
        out_shape=(H, W),
        transform=transform,
        fill=0,
        all_touched=True
    )

    return np.vstack(arrays), inside_mask, H, W, transform, crs

# ─────────────────────────────────────────────
# BUILD TRAINING DATA
# ─────────────────────────────────────────────
print("\n📚 Building training dataset (2014–2024)...\n")

X_all, y_all = [], []

for year in TRAIN_YEARS:
    mask_fp = os.path.join(MASK_DIR, f"{year}_mask.tif")
    if not os.path.exists(mask_fp):
        continue

    features_stack, inside_mask, H, W, _, _ = load_features(year)

    with rasterio.open(mask_fp) as src:
        labels = src.read(1)

    X = features_stack.reshape(features_stack.shape[0], -1).T
    y = labels.flatten()
    inside = inside_mask.flatten() == 1

    valid = (y > 0) & inside
    X_all.append(X[valid])
    y_all.append(y[valid])

X_all = np.vstack(X_all)
y_all = np.hstack(y_all)

# ─────────────────────────────────────────────
# BALANCE CLASSES
# ─────────────────────────────────────────────
print("⚖️ Balancing classes...\n")

X_bal, y_bal = [], []

for cls in [1, 2, 3]:
    idx = np.where(y_all == cls)[0]
    X_c, y_c = X_all[idx], y_all[idx]
    X_res, y_res = resample(
        X_c, y_c,
        n_samples=min(60000, len(y_c)),
        random_state=42
    )
    X_bal.append(X_res)
    y_bal.append(y_res)

X_bal = np.vstack(X_bal)
y_bal = np.hstack(y_bal)

print("✔ Training samples:", X_bal.shape[0])

# ─────────────────────────────────────────────
# TRAIN RANDOM FOREST
# ─────────────────────────────────────────────
print("\n🌳 Training RandomForest...\n")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_bal, y_bal)

print("✅ Model trained")

# ─────────────────────────────────────────────
# PREDICT OLD YEARS (WITH BOUNDARY MASK)
# ─────────────────────────────────────────────
print("\n🕰 Predicting 1988–2013...\n")

for year in tqdm(PREDICT_YEARS):
    features_stack, inside_mask, H, W, transform, crs = load_features(year)
    Xp = features_stack.reshape(features_stack.shape[0], -1).T

    preds = rf.predict(Xp)
    pred_mask = preds.reshape(H, W).astype("uint8")

    # 🚨 FORCE OUTSIDE CITY = 0 (BLACK)
    pred_mask[inside_mask == 0] = 0

    out_fp = os.path.join(OUT_DIR, f"{year}_mask_predicted.tif")

    meta = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform
    }

    with rasterio.open(out_fp, "w", **meta) as dst:
        dst.write(pred_mask, 1)

print("\n🎉 All predictions complete!")
