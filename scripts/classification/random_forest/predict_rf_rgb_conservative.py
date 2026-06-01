import os
import rasterio
import geopandas as gpd
import numpy as np
import joblib
from rasterio import features
from tqdm import tqdm

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT = r"C:\Users\kiran\Downloads"
MODEL_FP = os.path.join(ROOT, "saved_models", "rf_rgb_conservative.joblib")
OUT_DIR = os.path.join(ROOT, "final_classified_masks", "predictions_rgb_conservative")
os.makedirs(OUT_DIR, exist_ok=True)

BOUNDARY_FP = os.path.join(ROOT, "BENGALURU (1).geojson")

PREDICT_YEARS = list(range(1988, 2014))

# ─────────────────────────────────────────────
# LOAD BOUNDARY & MODEL
# ─────────────────────────────────────────────
boundary = gpd.read_file(BOUNDARY_FP)
rf_model = joblib.load(MODEL_FP)

# ─────────────────────────────────────────────
# LOAD RGB ONLY
# ─────────────────────────────────────────────
def load_rgb(year):
    rgb_fp = os.path.join(ROOT, f"{year}_RGBNIR_norm.tif")

    with rasterio.open(rgb_fp) as src:
        rgb = src.read([1, 2, 3])
        rgb = np.nan_to_num(rgb, nan=0)

        H, W = src.height, src.width
        transform = src.transform
        crs = src.crs

    boundary_pr = boundary.to_crs(crs)
    inside_mask = features.rasterize(
        [(g, 1) for g in boundary_pr.geometry],
        out_shape=(H, W),
        transform=transform,
        fill=0,
        all_touched=True
    )

    return rgb, inside_mask, H, W, transform, crs

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
print("🕰 Running predictions (RGB only)...")

for year in tqdm(PREDICT_YEARS):
    rgb, inside_mask, H, W, transform, crs = load_rgb(year)

    Xp = rgb.reshape(3, -1).T
    preds = rf_model.predict(Xp)

    pred_mask = preds.reshape(H, W).astype("uint8")
    pred_mask[inside_mask == 0] = 0

    out_fp = os.path.join(OUT_DIR, f"{year}_mask_predicted.tif")

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

print("🚀 ALL PREDICTIONS COMPLETED 🚀")
