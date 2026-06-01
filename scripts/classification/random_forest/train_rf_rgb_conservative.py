import os
import rasterio
import geopandas as gpd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from rasterio import features
from tqdm import tqdm

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT = r"C:\Users\kiran\Downloads"
MASK_DIR = ROOT
MODEL_DIR = os.path.join(ROOT, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

BOUNDARY_FP = os.path.join(ROOT, "BENGALURU (1).geojson")

TRAIN_YEARS = list(range(2014, 2025))

# ─────────────────────────────────────────────
# LOAD BOUNDARY
# ─────────────────────────────────────────────
boundary = gpd.read_file(BOUNDARY_FP)

# ─────────────────────────────────────────────
# RANDOM FOREST (CONSERVATIVE)
# ─────────────────────────────────────────────
rf_conservative = RandomForestClassifier(
    n_estimators=250,
    max_depth=18,
    min_samples_leaf=15,
    n_jobs=-1,
    random_state=42
)

# ─────────────────────────────────────────────
# LOAD RGB FEATURES ONLY
# ─────────────────────────────────────────────
def load_rgb(year):
    rgb_fp = os.path.join(ROOT, f"Landsat_{year}_RGBNIR.tif")

    with rasterio.open(rgb_fp) as src:
        # Take ONLY RGB → bands 1,2,3
        rgb = src.read([1, 2, 3])
        rgb = np.nan_to_num(rgb, nan=0)

        H, W = src.height, src.width
        transform = src.transform
        crs = src.crs

    # Rasterize boundary
    boundary_pr = boundary.to_crs(crs)
    inside_mask = features.rasterize(
        [(g, 1) for g in boundary_pr.geometry],
        out_shape=(H, W),
        transform=transform,
        fill=0,
        all_touched=True
    )

    return rgb, inside_mask, H, W

# ─────────────────────────────────────────────
# BUILD TRAINING DATA
# ─────────────────────────────────────────────
X_all, y_all = [], []

print("📚 Building training dataset...")

for year in tqdm(TRAIN_YEARS):
    mask_fp = os.path.join(MASK_DIR, f"{year}_mask.tif")
    if not os.path.exists(mask_fp):
        continue

    rgb, inside_mask, H, W = load_rgb(year)

    with rasterio.open(mask_fp) as src:
        labels = src.read(1)

    X = rgb.reshape(3, -1).T
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

# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────
print("🌳 Training Conservative RF (RGB only)...")
rf_conservative.fit(X_bal, y_bal)

# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────
model_fp = os.path.join(MODEL_DIR, "rf_rgb_conservative.joblib")
joblib.dump(rf_conservative, model_fp)

print(f"✅ Model saved at: {model_fp}")
