import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
from scipy.ndimage import distance_transform_edt

# ── INPUT PATHS
BOUNDARY_FP = r"C:\Users\kiran\Desktop\main el\BENGALURU (1).geojson"
BASE_FP     = r"C:\Users\kiran\Downloads\Sentinel_RGBNIR_2022.tif"
NDVI_FP     = r"C:\Users\kiran\Downloads\Sentinel_NDVI_2022.tif"
NDWI_FP     = r"C:\Users\kiran\Downloads\Sentinel_NDWI_2022.tif"
NDBI_FP     = r"C:\Users\kiran\Downloads\Sentinel_NDBI_2022.tif"

# OSM Layers you want to use
FILES = {
    "buildings": "gis_osm_buildings_a_free_1.shp",
    "roads": "gis_osm_roads_free_1.shp",
    "transport": "gis_osm_transport_a_free_1.shp",
    "railways": "gis_osm_railways_free_1.shp",
    "traffic": "gis_osm_traffic_a_free_1.shp",
    "landuse": "gis_osm_landuse_a_free_1.shp",
    "pois": "gis_osm_pois_free_1.shp",
    "natural": "gis_osm_natural_free_1.shp",
    "water": "gis_osm_water_a_free_1.shp",
    "waterways": "gis_osm_waterways_free_1.shp",
}

ROOT = r"C:\Users\kiran\Downloads\southern-zone-220101-free.shp\\"
for k in FILES:
    FILES[k] = ROOT + FILES[k]

OUTPUT_FP = r"C:\Users\kiran\Desktop\main el\Bengaluru_mask_updated.tif"

# ── LOAD BOUNDARY
boundary = gpd.read_file(BOUNDARY_FP)

# ── READ BASE RASTER FOR TRANSFORM AND CRS
with rasterio.open(BASE_FP) as src:
    meta = src.meta.copy()
    shape_hw = (src.height, src.width)
    transform = src.transform
    crs = src.crs

boundary = boundary.to_crs(crs)
inside_mask = features.rasterize(
    [(geom, 1) for geom in boundary.geometry],
    out_shape=shape_hw,
    transform=transform,
    fill=0,
    all_touched=True
)

# ── READ SPECTRAL INDICES
ndvi = rasterio.open(NDVI_FP).read(1)
ndwi = rasterio.open(NDWI_FP).read(1)
ndbi = rasterio.open(NDBI_FP).read(1)

# ── FINAL LABELS INIT → ALL UNKNOWN = 255
final = np.full(shape_hw, 255, dtype="uint8")

# Outside region = 0
final[inside_mask == 0] = 0
inside = (inside_mask == 1)

# ──  Built-up from ALL urban layers
urban_layers = ["buildings","roads","transport","railways","traffic","landuse","pois"]

for layer in urban_layers:
    gdf = gpd.read_file(FILES[layer]).to_crs(crs)
    gdf = gdf[gdf.geometry.notnull()]
    blr_clip = gdf.overlay(boundary, how="intersection")
    urban_raster = features.rasterize(
        [(geom, 1) for geom in blr_clip.geometry],
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        all_touched=True
    )
    final[inside & (urban_raster == 1) & (urban_raster == 1)] = 3  # mark built-up

# ── Water from OSM water + waterways
water_layers = ["water","waterways","natural"]

for layer in water_layers:
    gdf = gpd.read_file(FILES[layer]).to_crs(crs)
    gdf = gdf[gdf.geometry.notnull()]

    blr_clip = gdf.overlay(boundary, how="intersection")

    # If it's natural layer, filter only water-like features if fclass exists
    if "fclass" in blr_clip.columns:
        blr_clip = blr_clip[blr_clip["fclass"].isin(["water","river","reservoir","lake","pond","basin"])]

    water_raster = features.rasterize(
        [(geom, 1) for geom in blr_clip.geometry],
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        all_touched=True
    ).astype("uint8")

    # label water
    final[inside & (water_raster == 1)] = 1

# ── Fill spectral water where OSM missed
unknown = (final == 255) & inside
final[unknown & (ndwi >= 0.3)] = 1

# ── Fill vegetation from NDVI
unknown = (final == 255) & inside
final[unknown & (ndvi >= 0.5)] = 2

# ── Remaining unlabeled urban via NDBI
unknown = (final == 255) & inside
final[unknown & (ndbi >= 0.1)] = 3

# ── NEAREST NEIGHBOUR FILL FOR ANYTHING STILL UNKNOWN
if np.any(final == 255):
    print("NN fill for unknown pixels running...")
    mask_unk = (final == 255)
    _, (ni, nj) = distance_transform_edt(mask_unk, return_indices=True)
    final[mask_unk] = final[ni[mask_unk], nj[mask_unk]]

final[inside_mask == 0] = 0  # enforce background

# SAVE OUTPUT
meta.update({"count": 1, "dtype": "uint8", "crs": crs, "transform": transform})
with rasterio.open(OUTPUT_FP, "w", **meta) as dst:
    dst.write(final, 1)

print("Done ✅")
print("Classes present:", np.unique(final))
