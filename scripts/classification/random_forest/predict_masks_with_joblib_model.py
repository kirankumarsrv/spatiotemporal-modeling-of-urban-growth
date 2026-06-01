import os
import joblib
import rasterio
import numpy as np

DOWNLOADS = r"C:\Users\kiran\Downloads"

MODEL_PATH = os.path.join(DOWNLOADS, "rf_predicted_masks", "rf_bangalore_model.pkl")
OUTPUT_FOLDER = os.path.join(DOWNLOADS, "rf_predicted_masks", "predictions")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

years_to_predict = range(1988, 2014)  # 1988–2013


def load_band(filepath):
    """Loads a single-band or multi-band raster and returns (array, metadata)"""
    src = rasterio.open(filepath)
    arr = src.read()  # shape: (bands, H, W)
    return arr, src


def get_year_files(year):
    """Returns filepaths for NDVI, NDBI, NDWI, RGBNIR for that year"""

    patterns = {
        "ndvi": f"Landsat_{year}_NDVI.tif",
        "ndbi": f"Landsat_{year}_NDBI.tif",
        "ndwi": f"Landsat_{year}_NDWI.tif",
        "rgbnir": f"Landsat_{year}_RGBNIR.tif",
    }

    paths = {}
    for key, filename in patterns.items():
        path = os.path.join(DOWNLOADS, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        paths[key] = path

    return paths

print("📥 Loading model (mmap_mode='r')...")
model = joblib.load(MODEL_PATH, mmap_mode='r')
print("✅ Model loaded (memory-mapped).")
# Load model
# print("📥 Loading model...")
# model = joblib.load(MODEL_PATH)
# print("✅ Model loaded!")


for year in years_to_predict:
    print(f"\n➡ Predicting for {year}...")

    # Get all file paths
    files = get_year_files(year)

    # Load all bands
    ndvi, src_ref = load_band(files["ndvi"])
    ndbi, _ = load_band(files["ndbi"])
    ndwi, _ = load_band(files["ndwi"])
    rgbnir, _ = load_band(files["rgbnir"])  # might be 1 or 3+ bands

    # Stack the 4 rasters as bands
    # Ensure each one is "1 × H × W"
    stack = np.vstack([
        ndvi,
        ndbi,
        ndwi,
        rgbnir[0:1]  # use only the first band of RGBNIR for consistency
    ]).astype("float32")

    H, W = stack.shape[1], stack.shape[2]

    # Flatten to pixels × features
    X = stack.reshape(stack.shape[0], -1).T

    print(f"  • Running model prediction on {X.shape[0]} pixels...")
    y_pred = model.predict(X)

    mask = y_pred.reshape(H, W).astype("uint8")

    out_path = os.path.join(OUTPUT_FOLDER, f"{year}_mask_predicted.tif")

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype="uint8",
        crs=src_ref.crs,
        transform=src_ref.transform,
    ) as dst:
        dst.write(mask, 1)

    print(f"  ✔ Saved: {out_path}")

print("\n🎉 All predictions completed! Time to relax — even RandomForest needs a chai break ☕.")
