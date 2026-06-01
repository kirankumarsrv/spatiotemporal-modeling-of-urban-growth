import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import box

# 1. Load Sentinel image to get its real bounds and CRS
with rasterio.open(r"C:\Users\kiran\Desktop\main el\Sentinel_RGBNIR_2024.tif") as src:
    bounds = src.bounds
    transform = src.transform
    crs = src.crs
    shape = (src.height, src.width)
    meta = src.meta.copy()

# 2. Convert Sentinel bounds into a shapely polygon box
sentinel_bbox = gpd.GeoDataFrame(
    {"geometry": [box(bounds.left, bounds.bottom, bounds.right, bounds.top)]},
    crs=crs
)

# 3. Load Bengaluru boundary and reproject into Sentinel CRS
boundary = gpd.read_file(r"C:\Users\kiran\Desktop\main el\BENGALURU (1).geojson")
boundary = boundary.to_crs(crs)

# 4. Load southern zone buildings and reproject into Sentinel CRS
buildings = gpd.read_file(r"C:\Users\kiran\Downloads\southern-zone-240101-free.shp\gis_osm_buildings_a_free_1.shp")
buildings = buildings.to_crs(crs)

# 5. Clip buildings → Boundary → Sentinel BBox
bengaluru_buildings = gpd.overlay(buildings, boundary, how="intersection")
bengaluru_buildings = gpd.overlay(bengaluru_buildings, sentinel_bbox, how="intersection")

print("Buildings found in boundary:", len(bengaluru_buildings))

# 6. Rasterize if buildings exist
if len(bengaluru_buildings) == 0:
    print("❌ No buildings overlap the Sentinel extent!")
else:
    mask = features.rasterize(
        [(geom, 4) for geom in bengaluru_buildings.geometry],
        out_shape=shape,
        transform=transform
    )

    unique_vals = set(mask.flatten())
    print("Mask pixel values present:", unique_vals)

    # 7. Update metadata to single-band mask
    meta.update({"count": 1, "dtype": "uint8", "crs": crs, "transform": transform})

    # 8. Save labeled mask
    out_path = r"C:\Users\kiran\Desktop\main el\Bengaluru_builtup_mask.tif"
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask.astype("uint8"), 1)

    print("✅ Mask saved successfully to:", out_path)
