import os
import glob
import re
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# ─────────────────────────────────────────────
# ROOT PATHS
# ─────────────────────────────────────────────
PRED_ROOT = r"C:\Users\kiran\Downloads\final_classified_masks\predictions"
OUT_ROOT  = r"C:\Users\kiran\Downloads\final_classified_masks\comparison_visualizations"
os.makedirs(OUT_ROOT, exist_ok=True)

# ─────────────────────────────────────────────
# DECLARED EXPERIMENT SPACE (AUTHORITATIVE)
# ─────────────────────────────────────────────
FEATURE_SETS = [
    "fs_1_full",
    "fs_2_no_spatial",
    "fs_3_indices_only",
    "fs_4_rgbnir_only",
    "fs_5_rgbnir_spatial"
]

RF_MODELS = [
    "rf_1_generalist",
    "rf_2_shallow",
    "rf_3_deep",
    "rf_4_sqrt_features",
    "rf_5_conservative"
]

# ─────────────────────────────────────────────
# COLORMAP (CONSISTENT)
# ─────────────────────────────────────────────
cmap = colors.ListedColormap([
    "#000000",  # 0 outside
    "#1f78b4",  # 1 water
    "#33a02c",  # 2 vegetation
    "#fdbf6f",  # 3 built-up
])
norm = colors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

# ─────────────────────────────────────────────
# HELPER: EXPORT PNG
# ─────────────────────────────────────────────
def save_visual(mask, title, out_png):
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap=cmap, norm=norm, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()

# ─────────────────────────────────────────────
# 1️⃣ RF-ONLY MODELS
# ─────────────────────────────────────────────
print("\n🔍 Processing RF-only experiments...\n")

for rf in RF_MODELS:
    rf_dir = os.path.join(PRED_ROOT, rf)
    if not os.path.isdir(rf_dir):
        continue

    print(f"🧪 Found RF-only: {rf}")

    for tif in glob.glob(os.path.join(rf_dir, "*.tif")):
        fname = os.path.basename(tif)
        match = re.search(r"(19\d{2}|20\d{2})", fname)
        if not match:
            continue

        year = match.group(1)
        year_dir = os.path.join(OUT_ROOT, year)
        os.makedirs(year_dir, exist_ok=True)

        out_png = os.path.join(year_dir, f"{rf}.png")

        with rasterio.open(tif) as src:
            mask = src.read(1)

        save_visual(mask, f"{rf} — {year}", out_png)

# ─────────────────────────────────────────────
# 2️⃣ FEATURE-SET + RF COMBINATIONS
# ─────────────────────────────────────────────
print("\n🔍 Processing Feature-set + RF experiments...\n")

for fs in FEATURE_SETS:
    fs_dir = os.path.join(PRED_ROOT, fs)
    if not os.path.isdir(fs_dir):
        continue

    print(f"📂 Feature set detected: {fs}")

    for rf in RF_MODELS:
        rf_dir = os.path.join(fs_dir, rf)
        if not os.path.isdir(rf_dir):
            continue

        exp_name = f"{fs}_{rf}"
        print(f"   🧪 {exp_name}")

        for tif in glob.glob(os.path.join(rf_dir, "*.tif")):
            fname = os.path.basename(tif)
            match = re.search(r"(19\d{2}|20\d{2})", fname)
            if not match:
                continue

            year = match.group(1)
            year_dir = os.path.join(OUT_ROOT, year)
            os.makedirs(year_dir, exist_ok=True)

            out_png = os.path.join(year_dir, f"{exp_name}.png")

            with rasterio.open(tif) as src:
                mask = src.read(1)

            save_visual(mask, f"{exp_name} — {year}", out_png)

print("\n✅ ALL visualizations exported successfully!")
