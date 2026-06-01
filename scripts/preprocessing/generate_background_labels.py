import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import shape, mapping
import numpy as np

# ── Step 1: read Bengaluru boundary
boundary = gpd.read_file(r"C:\Users\kiran\Desktop\main el\BENGALURU (1).geojson")

# ── Step 2: open the Sentinel RGB-NIR image (canvas)
with rasterio.open(r"C:\Users\kiran\Downloads\Sentinel_RGBNIR_2022.tif") as src:
    meta = src.meta.copy()
    h, w = src.height, src.width
    transform = src.transform
    crs = src.crs

canvas_shape = (h, w)

# ── Step 3: reproject boundary to Sentinel CRS if needed
boundary = boundary.to_crs(crs)

# Rasterize the boundary itself → 1 = inside city, 0 = outside city
boundary_mask = features.rasterize(
    [(geom, 1) for geom in boundary.geometry],
    out_shape=canvas_shape,
    transform=transform,
    fill=0,  # explicit
    all_touched=True
)

# ── Step 4: read South India building footprints
buildings = gpd.read_file(r"C:\Users\kiran\Downloads\southern-zone-220101-free.shp\gis_osm_buildings_a_free_1.shp")
buildings = buildings.to_crs(crs)

# Clip buildings to Bengaluru polygon
bengaluru_buildings = gpd.overlay(buildings, boundary, how="intersection")

print("✔ Total building polygons after clipping:", len(bengaluru_buildings))

# Rasterize clipped buildings → value 4
built_raster = features.rasterize(
    [(geom, 4) for geom in bengaluru_buildings.geometry],
    out_shape=canvas_shape,
    transform=transform,
    fill=0,
    all_touched=True
)

# ── Step 5: read NDWI and NDVI rasters
with rasterio.open(r"C:\Users\kiran\Downloads\Sentinel_NDWI_2022.tif") as n1:
    ndwi = n1.read(1)

with rasterio.open(r"C:\Users\kiran\Downloads\Sentinel_NDVI_2022.tif") as n2:
    ndvi = n2.read(1)

assert ndwi.shape == built_raster.shape, "❌ NDWI shape mismatch!"
assert ndvi.shape == built_raster.shape, "❌ NDVI shape mismatch!"

# ── Step 6: build final labels raster
final = np.zeros(canvas_shape, dtype="uint8")

WATER_THR = 0.3
VEG_THR = 0.5

for i in range(h):
    for j in range(w):
        if boundary_mask[i, j] == 0:
            final[i, j] = 0  # outside city
        else:
            if built_raster[i, j] == 4:
                final[i, j] = 4  # building
            elif ndwi[i, j] >= WATER_THR:
                final[i, j] = 1  # water
            elif ndvi[i, j] >= VEG_THR:
                final[i, j] = 2  # vegetation
            else:
                final[i, j] = 3  # open/others

# ── Step 7: save the final mask
meta.update({"count": 1, "dtype": "uint8", "crs": crs, "transform": transform})

with rasterio.open(r"C:\Users\kiran\Desktop\main el\Bengaluru_labeled_mask_2022_FINAL.tif", "w", **meta) as dst:
    dst.write(final, 1)

print("✅ Final mask saved!")
print("Unique labels present:", np.unique(final))
print("Built-up pixel count:", (final == 4).sum())
print("Outside city pixel count:", (final == 0).sum())
