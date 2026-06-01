import os
import re
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# ROOT FOLDER (where RGBNIR TIFs already exist)
# ─────────────────────────────────────────────
ROOT = r"C:\Users\kiran\Downloads"

print("\n🎨 Creating RGB visualizations next to RGBNIR TIF files...\n")

# ─────────────────────────────────────────────
# Percentile stretch for good visualization
# ─────────────────────────────────────────────
def stretch(band, p_low=2, p_high=98):
    lo, hi = np.percentile(band, (p_low, p_high))
    band = np.clip(band, lo, hi)
    return (band - lo) / (hi - lo + 1e-6)

count = 0

for fname in os.listdir(ROOT):
    match = re.match(r"Landsat_(\d{4})_RGBNIR\.tif$", fname)
    if not match:
        continue

    year = match.group(1)
    tif_fp = os.path.join(ROOT, fname)
    png_fp = os.path.join(ROOT, f"Landsat_{year}_RGBNIR.png")

    print(f"📅 {year} → visualizing {fname}")

    with rasterio.open(tif_fp) as src:
        # Assume band order: R, G, B, NIR
        r = src.read(1).astype("float32")
        g = src.read(2).astype("float32")
        b = src.read(3).astype("float32")

    # Stretch for display
    r = stretch(r)
    g = stretch(g)
    b = stretch(b)

    rgb = np.dstack([r, g, b])

    # Save PNG beside the TIF
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title(f"Bengaluru RGB – {year}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_fp, dpi=150)
    plt.close()

    print(f"✅ Saved → {png_fp}")
    count += 1

print("\n────────────────────────────────────")
print(f"🎉 Done. Total RGB PNGs created: {count}")
print("📌 PNGs are saved NEXT TO their RGBNIR TIF files.")
