import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import box
import numpy as np
import matplotlib.pyplot as plt

# ── Step 1: Load canvas to get grid, bounds, CRS
with rasterio.open(r"C:\Users\kiran\Desktop\main el\Sentinel_RGBNIR_2024.tif") as src:
    meta = src.meta.copy()
    shape = (src.height, src.width)
    transform = src.transform
    crs = src.crs
    bounds = src.bounds

# ── Step 2: Load Bengaluru boundary + reproject to canvas CRS
boundary = gpd.read_file(r"C:\Users\kiran\Desktop\main el\BENGALURU (1).geojson")
boundary = boundary.to_crs(crs)

# ── Step 3: Load southern buildings + reproject
buildings = gpd.read_file(
    r"C:\Users\kiran\Downloads\southern-zone-240101-free.shp\gis_osm_buildings_a_free_1.shp"
)
buildings = buildings.to_crs(crs)

# ── Step 4: Clip buildings to boundary and Sentinel extent
sentinel_bbox = gpd.GeoDataFrame(
    {"geometry": [box(bounds.left, bounds.bottom, bounds.right, bounds.top)]}, crs=crs
)
bengaluru_buildings = gpd.overlay(buildings, boundary, how="intersection")
bengaluru_buildings = gpd.overlay(bengaluru_buildings, sentinel_bbox, how="intersection")

print("✔ Buildings inside Bengaluru + image extent:", len(bengaluru_buildings))

# ── Step 5: Rasterize buildings → label 4
built_mask = features.rasterize(
    [(geom, 4) for geom in bengaluru_buildings.geometry],
    out_shape=shape,
    transform=transform
)

# Preview (optional debug)
plt.figure()
plt.imshow(built_mask, vmin=0, vmax=4)
plt.title("Built=4 mask debug view")
plt.show()

# ── Step 6: Load NDWI & NDVI rasters
with rasterio.open(r"C:\Users\kiran\Downloads\Sentinel_NDWI_2024.tif") as n1:
    ndwi = n1.read(1)

with rasterio.open(r"C:\Users\kiran\Downloads\Sentinel_NDVI_2024.tif") as n2:
    ndvi = n2.read(1)

# Ensure alignment
assert ndwi.shape == built_mask.shape, "NDWI size mismatch!"
assert ndvi.shape == built_mask.shape, "NDVI size mismatch!"

# ── Step 7: Create final label array with priority stack
final_labels = np.zeros(shape, dtype="uint8")

WATER_THR = 0.3
VEG_THR = 0.5

h, w = shape
for i in range(h):
    for j in range(w):
        if built_mask[i, j] == 4:
            final_labels[i, j] = 4
        elif ndwi[i, j] >= WATER_THR:
            final_labels[i, j] = 1
        elif ndvi[i, j] >= VEG_THR:
            final_labels[i, j] = 2
        else:
            final_labels[i, j] = 3

# ── Step 8: Save final label mask
meta.update({"count": 1, "dtype": "uint8", "crs": crs, "transform": transform})

out_path = r"C:\Users\kiran\Desktop\main el\Bengaluru_labeled_mask.tif"
with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(final_labels, 1)

print("✅ Final labeled mask saved to:", out_path)
print("Mask contains pixel values:", set(final_labels.flatten()))
