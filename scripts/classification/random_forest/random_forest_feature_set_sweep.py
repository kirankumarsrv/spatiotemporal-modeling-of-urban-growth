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
# FEATURE SETS
# ─────────────────────────────────────────────
FEATURE_SETS = [
    "fs_1_full",
    "fs_2_no_spatial",
    "fs_3_indices_only",
    "fs_4_rgbnir_only",
    "fs_5_rgbnir_spatial"
]

# ─────────────────────────────────────────────
# RANDOM FOREST MODELS
# ─────────────────────────────────────────────
rf_models = {
    "rf_1_generalist": RandomForestClassifier(
        n_estimators=300, max_depth=25, min_samples_leaf=5,
        n_jobs=-1, random_state=42
    ),
    "rf_2_shallow": RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=10,
        n_jobs=-1, random_state=42
    ),
    "rf_3_deep": RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42
    ),
    "rf_4_sqrt_features": RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=42
    ),
    "rf_5_conservative": RandomForestClassifier(
        n_estimators=250, max_depth=18, min_samples_leaf=15,
        n_jobs=-1, random_state=42
    )
}

# ─────────────────────────────────────────────
# LOAD FEATURES (BASED ON FEATURE SET)
# ─────────────────────────────────────────────
def load_features(year, feature_set):
    files = {
        "rgbnir": f"Landsat_{year}_RGBNIR.tif",
        "ndvi":   f"Landsat_{year}_NDVI.tif",
        "ndwi":   f"Landsat_{year}_NDWI.tif",
        "ndbi":   f"Landsat_{year}_NDBI.tif",
    }

    arrays = []
    H = W = None

    # RGBNIR
    if feature_set in ["fs_1_full", "fs_2_no_spatial", "fs_4_rgbnir_only", "fs_5_rgbnir_spatial"]:
        with rasterio.open(os.path.join(ROOT, files["rgbnir"])) as src:
            arrays.append(np.nan_to_num(src.read(), nan=0))
            H, W = src.height, src.width
            transform = src.transform
            crs = src.crs

    # Indices
    if feature_set in ["fs_1_full", "fs_2_no_spatial", "fs_3_indices_only"]:
        for k in ["ndvi", "ndwi", "ndbi"]:
            with rasterio.open(os.path.join(ROOT, files[k])) as src:
                arrays.append(np.nan_to_num(src.read(), nan=0))
                if H is None:
                    H, W = src.height, src.width
                    transform = src.transform
                    crs = src.crs

    # Spatial features
    if feature_set in ["fs_1_full", "fs_5_rgbnir_spatial"]:
        rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        arrays.append(rows[None, :, :])
        arrays.append(cols[None, :, :])

    # Rasterized boundary
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
# MAIN LOOP: FEATURE SET × MODEL
# ─────────────────────────────────────────────
for feature_set in FEATURE_SETS:

    print(f"\n🧩 FEATURE SET: {feature_set}\n")

    # ── BUILD TRAINING DATA
    X_all, y_all = [], []

    for year in TRAIN_YEARS:
        mask_fp = os.path.join(MASK_DIR, f"{year}_mask.tif")
        if not os.path.exists(mask_fp):
            continue

        features_stack, inside_mask, H, W, _, _ = load_features(year, feature_set)

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

    # ── BALANCE CLASSES
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

    # ── TRAIN & PREDICT FOR EACH RF
    for model_name, rf in rf_models.items():

        print(f"🌳 Training {feature_set} → {model_name}")
        rf.fit(X_bal, y_bal)

        model_out_dir = os.path.join(OUT_DIR, feature_set, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        print(f"🕰 Predicting with {model_name}...")

        for year in tqdm(PREDICT_YEARS):
            features_stack, inside_mask, H, W, transform, crs = load_features(year, feature_set)
            Xp = features_stack.reshape(features_stack.shape[0], -1).T

            preds = rf.predict(Xp)
            pred_mask = preds.reshape(H, W).astype("uint8")
            pred_mask[inside_mask == 0] = 0

            out_fp = os.path.join(model_out_dir, f"{year}_mask_predicted.tif")

            with rasterio.open(
                out_fp, "w",
                driver="GTiff",
                height=H,
                width=W,
                count=1,
                dtype="uint8",
                crs=crs,
                transform=transform
            ) as dst:
                dst.write(pred_mask, 1)

        print(f"✅ Done: {feature_set} → {model_name}")

print("\n🚀 ALL FEATURE SETS & MODELS COMPLETED 🚀")
