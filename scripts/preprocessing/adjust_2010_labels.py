import rasterio
import numpy as np
from scipy.ndimage import distance_transform_edt

# -----------------------------
# FILE PATHS
# -----------------------------
mask_2005 = r"C:\Users\kiran\Downloads\final_classified_masks\predictions_rgb_conservative\2005_mask_predicted.tif"
mask_2010 = r"C:\Users\kiran\Downloads\final_classified_masks\predictions_rgb_conservative\2010_mask_predicted.tif"
mask_2016 = r"C:\Users\kiran\Downloads\2016_mask (2).tif"

out_2010 = r"C:\Users\kiran\Downloads\final_classified_masks\2010_mask_adjusted.tif"

# -----------------------------
# LOAD MASKS
# -----------------------------
with rasterio.open(mask_2005) as src:
    m2005 = src.read(1)
    meta = src.meta.copy()

with rasterio.open(mask_2010) as src:
    m2010 = src.read(1)

with rasterio.open(mask_2016) as src:
    m2016 = src.read(1)

# -----------------------------
# LABEL DEFINITIONS
# -----------------------------
BACKGROUND = 0
WATER = 1
VEGETATION = 2
BUILTUP = 3

# -----------------------------
# STEP 1: PROTECT WATER
# -----------------------------
water_mask = (
    (m2005 == WATER) |
    (m2010 == WATER) |
    (m2016 == WATER)
)

# -----------------------------
# STEP 2: FORCE BUILT-UP CONTINUITY
# -----------------------------
builtup_mask = (
    (m2005 == BUILTUP) |
    (m2016 == BUILTUP)
)

# -----------------------------
# STEP 3: VEGETATION REDUCTION
# -----------------------------
veg_2010 = (m2010 == VEGETATION)

# Distance from built-up (closer veg converts first)
distance = distance_transform_edt(~builtup_mask)

# Normalize distance
distance_norm = distance / distance.max()

# Threshold controls how aggressive reduction is
# 0.35–0.45 is SAFE (start with 0.4)
VEG_REDUCTION_THRESHOLD = 0.40

convert_to_built = (
    veg_2010 &
    (distance_norm < VEG_REDUCTION_THRESHOLD)
)

# -----------------------------
# BUILD FINAL 2010 MASK
# -----------------------------
adjusted_2010 = m2010.copy()

# Apply conversions
adjusted_2010[convert_to_built] = BUILTUP

# Enforce built-up continuity
adjusted_2010[builtup_mask] = BUILTUP

# Restore water
adjusted_2010[water_mask] = WATER

# Enforce background
adjusted_2010[m2010 == BACKGROUND] = BACKGROUND

# -----------------------------
# SAVE RESULT
# -----------------------------
meta.update(dtype="uint8", count=1)

with rasterio.open(out_2010, "w", **meta) as dst:
    dst.write(adjusted_2010.astype("uint8"), 1)

print("✅ Adjusted 2010 mask saved successfully")
