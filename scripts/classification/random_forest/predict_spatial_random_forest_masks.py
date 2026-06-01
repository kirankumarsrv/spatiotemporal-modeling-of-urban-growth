# import os
# import rasterio
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.utils import resample
# from tqdm import tqdm

# # ─────────────────────────────────────────────
# # PATHS
# # ─────────────────────────────────────────────
# ROOT = r"C:\Users\kiran\Downloads"
# MASK_DIR = ROOT
# OUT_DIR = os.path.join(ROOT, "final_classified_masks", "predictions")
# os.makedirs(OUT_DIR, exist_ok=True)

# TRAIN_YEARS = list(range(2014, 2025))
# PREDICT_YEARS = list(range(1988, 2014))

# # ─────────────────────────────────────────────
# # FEATURE STACK FUNCTION
# # ─────────────────────────────────────────────
# def load_features(year):
#     files = {
#         "rgbnir": f"Landsat_{year}_RGBNIR.tif",
#         "ndvi":   f"Landsat_{year}_NDVI.tif",
#         "ndwi":   f"Landsat_{year}_NDWI.tif",
#         "ndbi":   f"Landsat_{year}_NDBI.tif",
#     }

#     arrays = []
#     for f in files.values():
#         with rasterio.open(os.path.join(ROOT, f)) as src:
#             arrays.append(np.nan_to_num(src.read(), nan=0))

#     with rasterio.open(os.path.join(ROOT, files["rgbnir"])) as src:
#         H, W = src.height, src.width

#     # Spatial coordinates
#     rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
#     arrays.append(rows[None, :, :])
#     arrays.append(cols[None, :, :])

#     return np.vstack(arrays), H, W

# # ─────────────────────────────────────────────
# # BUILD TRAINING DATA
# # ─────────────────────────────────────────────
# print("\n📚 Building training dataset (2014–2024)...\n")

# X_all, y_all = [], []

# for year in TRAIN_YEARS:
#     mask_fp = os.path.join(MASK_DIR, f"{year}_mask.tif")
#     if not os.path.exists(mask_fp):
#         continue

#     features, H, W = load_features(year)

#     with rasterio.open(mask_fp) as src:
#         labels = src.read(1)

#     X = features.reshape(features.shape[0], -1).T
#     y = labels.flatten()

#     valid = y > 0
#     X_all.append(X[valid])
#     y_all.append(y[valid])

# X_all = np.vstack(X_all)
# y_all = np.hstack(y_all)

# # ─────────────────────────────────────────────
# # BALANCE CLASSES
# # ─────────────────────────────────────────────
# print("⚖️ Balancing classes...\n")

# X_bal, y_bal = [], []

# for cls in [1, 2, 3]:
#     idx = np.where(y_all == cls)[0]
#     X_c, y_c = X_all[idx], y_all[idx]
#     X_res, y_res = resample(
#         X_c, y_c,
#         n_samples=min(60000, len(y_c)),
#         random_state=42
#     )
#     X_bal.append(X_res)
#     y_bal.append(y_res)

# X_bal = np.vstack(X_bal)
# y_bal = np.hstack(y_bal)

# print("✔ Training samples:", X_bal.shape[0])

# # ─────────────────────────────────────────────
# # TRAIN RANDOM FOREST
# # ─────────────────────────────────────────────
# print("\n🌳 Training RandomForest...\n")

# rf = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=25,
#     min_samples_leaf=5,
#     n_jobs=-1,
#     random_state=42
# )

# rf.fit(X_bal, y_bal)

# print("✅ Model trained")

# # ─────────────────────────────────────────────
# # PREDICT OLD YEARS
# # ─────────────────────────────────────────────
# print("\n🕰 Predicting 1988–2013...\n")

# for year in tqdm(PREDICT_YEARS):
#     features, H, W = load_features(year)
#     Xp = features.reshape(features.shape[0], -1).T

#     probs = rf.predict_proba(Xp)
#     preds = np.argmax(probs, axis=1) + 1  # classes 1–3

#     pred_mask = preds.reshape(H, W).astype("uint8")

#     out_fp = os.path.join(OUT_DIR, f"{year}_mask_predicted.tif")
#     with rasterio.open(
#         os.path.join(ROOT, f"Landsat_{year}_RGBNIR.tif")
#     ) as ref:
#         meta = ref.meta.copy()

#     meta.update(dtype="uint8", count=1)

#     with rasterio.open(out_fp, "w", **meta) as dst:
#         dst.write(pred_mask, 1)

# print("\n🎉 All predictions complete!")

# # ---------------------------------------------------------------------------------------
# import os
# import numpy as np
# import rasterio
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.utils.class_weight import compute_class_weight

# # ─────────────────────────────────────────────
# # 1️⃣ PATH CONFIG (MATCHES YOUR SYSTEM)
# # ─────────────────────────────────────────────
# DOWNLOADS = r"C:\Users\kiran\Downloads"

# # MASK_DIR = os.path.join(DOWNLOADS, "blr_yearly_masks")  # <-- YOUR masks
# MODEL_OUT = os.path.join(DOWNLOADS, "rf_landcover_model.joblib")

# YEARS = range(2014, 2025)

# # ─────────────────────────────────────────────
# # 2️⃣ COLLECT TRAINING DATA
# # ─────────────────────────────────────────────
# X_list = []
# y_list = []
# used_years = []

# print("\n🚀 Collecting training samples...\n")

# for year in YEARS:
#     print(f"📅 Year {year}")

#     rgb_fp  = os.path.join(DOWNLOADS, f"Landsat_{year}_RGBNIR.tif")
#     ndvi_fp = os.path.join(DOWNLOADS, f"Landsat_{year}_NDVI.tif")
#     ndwi_fp = os.path.join(DOWNLOADS, f"Landsat_{year}_NDWI.tif")
#     ndbi_fp = os.path.join(DOWNLOADS, f"Landsat_{year}_NDBI.tif")
#     mask_fp = os.path.join(DOWNLOADS, f"{year}_mask.tif")

#     required = [rgb_fp, ndvi_fp, ndwi_fp, ndbi_fp, mask_fp]
#     if not all(os.path.exists(p) for p in required):
#         print(f"❌ Missing files → skipped {required}")
#         continue

#     # ── Load RGBNIR
#     with rasterio.open(rgb_fp) as src:
#         rgbnir = src.read().reshape(4, -1).T
#         H, W = src.height, src.width

#     # ── Load indices
#     ndvi = np.nan_to_num(rasterio.open(ndvi_fp).read(1), nan=-1).ravel()
#     ndwi = np.nan_to_num(rasterio.open(ndwi_fp).read(1), nan=0).ravel()
#     ndbi = np.nan_to_num(rasterio.open(ndbi_fp).read(1), nan=-1).ravel()

#     # ── Load label mask
#     mask = rasterio.open(mask_fp).read(1).ravel()

#     # ── Keep only city pixels (mask != 0)
#     valid = mask != 0
#     if valid.sum() == 0:
#         print("⚠️ No valid city pixels → skipped")
#         continue

#     # ── Spatial features
#     rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
#     rows = rows.ravel()
#     cols = cols.ravel()

#     # ── Feature matrix
#     X = np.column_stack([
#         rgbnir[valid],
#         ndvi[valid],
#         ndwi[valid],
#         ndbi[valid],
#         rows[valid],
#         cols[valid]
#     ])

#     y = mask[valid]

#     X_list.append(X)
#     y_list.append(y)
#     used_years.append(year)

#     print(f"✅ Samples collected: {X.shape[0]}")

# # ─────────────────────────────────────────────
# # 3️⃣ SAFETY CHECK (NO MORE vstack ERROR)
# # ─────────────────────────────────────────────
# if len(X_list) == 0:
#     raise RuntimeError(
#         "❌ NO TRAINING DATA COLLECTED.\n"
#         "Check mask folder, filenames, and mask values."
#     )

# X = np.vstack(X_list)
# y = np.hstack(y_list)

# print("\n✅ Training years used:", used_years)
# print("📊 Total samples:", X.shape[0])
# print("📊 Classes:", np.unique(y))

# # ─────────────────────────────────────────────
# # 4️⃣ BALANCED RANDOM FOREST
# # ─────────────────────────────────────────────
# classes = np.unique(y)
# weights = compute_class_weight("balanced", classes=classes, y=y)
# class_weights = dict(zip(classes, weights))

# print("⚖️ Class weights:", class_weights)

# rf = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=25,
#     min_samples_leaf=20,
#     n_jobs=-1,
#     random_state=42,
#     class_weight=class_weights
# )

# print("\n🧠 Training Random Forest...")
# rf.fit(X, y)

# # ─────────────────────────────────────────────
# # 5️⃣ SAVE MODEL
# # ─────────────────────────────────────────────
# joblib.dump(rf, MODEL_OUT)

# print("\n💾 MODEL SAVED SUCCESSFULLY")
# print("📍", MODEL_OUT)
# print("🎉 Training complete!")
# ----------------------------------------------------------------------------------







import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
import joblib
import os

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
ROOT = r"C:\Users\kiran\Downloads"
MODEL_FP = os.path.join(ROOT, "rf_landcover_model.joblib")
BOUNDARY_FP = os.path.join(ROOT, "BENGALURU (1).geojson")

OUT_DIR = os.path.join(ROOT, "final_classified_masks", "predictions")
os.makedirs(OUT_DIR, exist_ok=True)

# Years to predict (old years)
YEARS = range(1988, 2014)

# Thresholds (Landsat-tuned)
NDWI_THR = 0.07
NDVI_THR = 0.18
NDBI_THR = 0.10

# ─────────────────────────────────────────
# LOAD MODEL + BOUNDARY
# ─────────────────────────────────────────
print("📦 Loading RandomForest model...")
rf = joblib.load(MODEL_FP)

print("🧭 Loading Bengaluru boundary...")
boundary = gpd.read_file(BOUNDARY_FP)

# ─────────────────────────────────────────
# HELPER: rasterize boundary
# ─────────────────────────────────────────
def get_inside_mask(rgbnir_fp, boundary_gdf):
    with rasterio.open(rgbnir_fp) as src:
        H, W = src.height, src.width
        transform = src.transform
        crs = src.crs

    boundary_pr = boundary_gdf.to_crs(crs)

    inside = features.rasterize(
        [(geom, 1) for geom in boundary_pr.geometry],
        out_shape=(H, W),
        transform=transform,
        fill=0,
        all_touched=True
    )

    return inside.astype(bool)

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
for year in YEARS:
    print(f"\n──────── YEAR {year} ────────")

    rgb_fp  = os.path.join(ROOT, f"Landsat_{year}_RGBNIR.tif")
    ndvi_fp = os.path.join(ROOT, f"Landsat_{year}_NDVI.tif")
    ndwi_fp = os.path.join(ROOT, f"Landsat_{year}_NDWI.tif")
    ndbi_fp = os.path.join(ROOT, f"Landsat_{year}_NDBI.tif")

    if not all(map(os.path.exists, [rgb_fp, ndvi_fp, ndwi_fp, ndbi_fp])):
        print("⚠️ Missing inputs → skipping")
        continue

    # ── Read rasters
    with rasterio.open(rgb_fp) as src:
        meta = src.meta.copy()
        rgbnir = src.read().reshape(4, -1).T
        H, W = src.height, src.width

    ndvi = np.nan_to_num(rasterio.open(ndvi_fp).read(1), nan=-1).ravel()
    ndwi = np.nan_to_num(rasterio.open(ndwi_fp).read(1), nan=0).ravel()
    ndbi = np.nan_to_num(rasterio.open(ndbi_fp).read(1), nan=-1).ravel()

    # ── Spatial features
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rows = rows.ravel()
    cols = cols.ravel()

    # ── Feature matrix (MUST match training)
    X = np.column_stack([rgbnir, ndvi, ndwi, ndbi, rows, cols])

    print("🤖 Running RandomForest prediction...")
    rf_pred = rf.predict(X)

    mask = rf_pred.reshape(H, W).astype("uint8")

    # ─────────────────────────────────────────
    # 🔧 HYBRID REFINEMENT (VERY IMPORTANT)
    # ─────────────────────────────────────────
    ndvi2 = ndvi.reshape(H, W)
    ndwi2 = ndwi.reshape(H, W)
    ndbi2 = ndbi.reshape(H, W)

    # Strong water correction
    mask[ndwi2 >= NDWI_THR] = 1

    # Strong vegetation correction
    mask[(ndvi2 >= NDVI_THR) & (ndwi2 < NDWI_THR)] = 2

    # Built-up correction
    mask[(ndbi2 >= NDBI_THR) & (ndvi2 < NDVI_THR)] = 3

    # ─────────────────────────────────────────
    # 🔒 FORCE OUTSIDE BENGALURU = 0
    # ─────────────────────────────────────────
    inside = get_inside_mask(rgb_fp, boundary)
    mask[~inside] = 0

    # ── Save
    out_fp = os.path.join(OUT_DIR, f"{year}_mask_predicted.tif")
    meta.update(count=1, dtype="uint8")

    with rasterio.open(out_fp, "w", **meta) as dst:
        dst.write(mask, 1)

    print("✅ Saved:", out_fp)
    print("Classes:", np.unique(mask))
