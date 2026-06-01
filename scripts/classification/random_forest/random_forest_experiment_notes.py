# import geopandas as gpd
# import rasterio
# from rasterio import features
# import numpy as np
# from scipy.ndimage import distance_transform_edt
# import os
# import re

# # ── 1. Set root paths (Downloads folder)
# ROOT = r"C:\Users\kiran\Downloads"
# OSM_ROOT = ROOT  # OSM folders are directly in Downloads

# print("\n🧾 Scanning available OSM extract folders...\n")
# osm_dirs = [d for d in os.listdir(OSM_ROOT) if os.path.isdir(os.path.join(OSM_ROOT, d)) and 'india-' in d and 'free.shp' in d]
# for d in osm_dirs:
#     print(" -", d)

# # ── 2. Load Bengaluru boundary region
# print("\n🧭 Loading Bengaluru boundary file...\n")
# boundary_fp = os.path.join(ROOT, "BENGALURU (1).geojson")
# boundary = gpd.read_file(boundary_fp)
# print("✔ Boundary loaded with geometry count:", len(boundary))


# # ── 3. Function to decode year from folder names like india-140101-free.shp
# def decode_year(folder_name):
#     # Look for 6-digit date chunks like 140101, 230101 etc.
#     nums = re.findall(r"(\d{2})(\d{2})(\d{2})", folder_name)
#     if nums:
#         y = nums[0][0]  # first two digits = year
#         year = 2000 + int(y)  # all are 20xx in your case
#         return year
#     return None

# # Mapping: year -> OSM folder path
# year_to_osm = {}
# for d in osm_dirs:
#     year = decode_year(d)
#     if year:
#         year_to_osm[year] = os.path.join(OSM_ROOT, d)

# print("\n🔎 OSM folders decoded to year mapping:\n")
# for y, fp in sorted(year_to_osm.items()):
#     print(f"  {y}: {fp}")

# # ── 4. Output dir (create in Downloads folder)
# out_mask_dir = os.path.join(ROOT, "test_masks")
# os.makedirs(out_mask_dir, exist_ok=True)

# print("\n🚀 Starting year-wise classification...\n")

# # ── 5. Loop over years
# for year in range(2018, 2025):
#     print("\n──────────────")
#     print(f"YEAR → {year}")
#     print("──────────────")

#     # Filenames for this year (all in Downloads folder)
#     base_fp = os.path.join(ROOT, f"Landsat_{year}_RGBNIR.tif")
#     ndvi_fp = os.path.join(ROOT, f"Landsat_{year}_NDVI.tif")
#     ndwi_fp = os.path.join(ROOT, f"Landsat_{year}_NDWI.tif")
#     ndbi_fp = os.path.join(ROOT, f"Landsat_{year}_NDBI.tif")
#     mask_out_fp = os.path.join(out_mask_dir, f"{year}_mask.tif")

#     # Check if required rasters exist
#     if not os.path.exists(base_fp):
#         print(f"❌ {base_fp} not found → skipping year {year}")
#         continue
#     if not (os.path.exists(ndvi_fp) and os.path.exists(ndwi_fp) and os.path.exists(ndbi_fp)):
#         print(f"❌ NDVI/NDWI/NDBI missing for {year} → skipping")
#         continue

#     # Check OSM availability
#     osm_year_dir = year_to_osm.get(year)
#     if osm_year_dir:
#         print("✅ OSM snapshot available for this year!")
#         print("📍 OSM folder:", osm_year_dir)
#     else:
#         print("⚠️ No OSM for this year – will use spectral-only classification")

#     # ── Read base raster to get canvas + CRS
#     with rasterio.open(base_fp) as src:
#         meta = src.meta.copy()
#         shape_hw = (src.height, src.width)
#         transform = src.transform
#         crs = src.crs

#     print("✔ Canvas size:", shape_hw)
#     print("✔ CRS:", crs)

#     # ── Reproject boundary to raster CRS
#     boundary_pr = boundary.to_crs(crs)

#     # ── Rasterize Bengaluru boundary (1 = inside, 0 = outside)
#     inside_mask = features.rasterize(
#         [(geom, 1) for geom in boundary_pr.geometry],
#         out_shape=shape_hw,
#         transform=transform,
#         fill=0,
#         all_touched=True
#     )

#     # ── Initialize final labels: 255 = unknown
#     final = np.full(shape_hw, 255, dtype="uint8")
#     final[inside_mask == 0] = 0  # outside city
#     inside = (inside_mask == 1)

#     # ── Load spectral indices
#     with rasterio.open(ndvi_fp) as src_ndvi:
#         ndvi = src_ndvi.read(1)
#     with rasterio.open(ndwi_fp) as src_ndwi:
#         ndwi = src_ndwi.read(1)
#     with rasterio.open(ndbi_fp) as src_ndbi:
#         ndbi = src_ndbi.read(1)
#     ndvi = np.nan_to_num(ndvi, nan=-1)
#     ndwi = np.nan_to_num(ndwi, nan=0)
#     ndbi = np.nan_to_num(ndbi, nan=-1)

#     print("✔ NDVI min/max:", float(ndvi.min()), float(ndvi.max()))
#     print("✔ NDWI min/max:", float(ndwi.min()), float(ndwi.max()))
#     print("✔ NDBI min/max:", float(ndbi.min()), float(ndbi.max()))

#     # ─────────────────────────────────────
#     #  A. OSM-based labels (if available)
#     # ─────────────────────────────────────
#     if osm_year_dir:
#         print("\n🗺 Applying OSM-based rasterization (built-up + water)...")

#         # Built-up: roads, buildings, transport, railways, traffic, landuse, pois
#         urban_layers = ["buildings", "roads", "transport", "railways", "traffic", "landuse", "pois"]
#         for layer in urban_layers:
#             layer_fp = os.path.join(osm_year_dir, f"gis_osm_{layer}_a_free_1.shp")
#             if not os.path.exists(layer_fp):
#                 layer_fp = os.path.join(osm_year_dir, f"gis_osm_{layer}_free_1.shp")
#             if os.path.exists(layer_fp):
#                 print(f"   → rasterizing urban layer: {layer}")
#                 gdf = gpd.read_file(layer_fp).to_crs(crs)
#                 if gdf.empty:
#                     continue
#                 blr_clip = gdf.overlay(boundary_pr, how="intersection")
#                 if blr_clip.empty:
#                     continue
#                 urban_r = features.rasterize(
#                     [(geom, 1) for geom in blr_clip.geometry if geom is not None],
#                     out_shape=shape_hw,
#                     transform=transform,
#                     fill=0,
#                     all_touched=True
#                 )
#                 final[inside & (urban_r == 1)] = 3  # built-up

#         # Water: water, waterways, water-like natural
#         water_layers = ["water", "waterways", "natural"]
#         for layer in water_layers:
#             layer_fp = os.path.join(osm_year_dir, f"gis_osm_{layer}_a_free_1.shp")
#             if not os.path.exists(layer_fp):
#                 layer_fp = os.path.join(osm_year_dir, f"gis_osm_{layer}_free_1.shp")
#             if os.path.exists(layer_fp):
#                 print(f"   → rasterizing water layer: {layer}")
#                 gdf = gpd.read_file(layer_fp).to_crs(crs)
#                 if gdf.empty:
#                     continue
#                 blr_clip = gdf.overlay(boundary_pr, how="intersection")
#                 if "fclass" in blr_clip.columns:
#                     blr_clip = blr_clip[blr_clip["fclass"].isin(
#                         ["water", "river", "reservoir", "lake", "pond", "basin"]
#                     )]
#                 if blr_clip.empty:
#                     continue
#                 water_r = features.rasterize(
#                     [(geom, 1) for geom in blr_clip.geometry if geom is not None],
#                     out_shape=shape_hw,
#                     transform=transform,
#                     fill=0,
#                     all_touched=True
#                 )
#                 final[inside & (water_r == 1)] = 1  # water

#     else:
#         print("🛈 Skipping OSM step (no snapshot for this year).")

#     # ─────────────────────────────────────
#     #  B. Spectral rules
#     # ─────────────────────────────────────
#     print("\n🖌 Applying spectral NDWI/NDVI/NDBI rules...")

#     # Thresholds
#     NDWI_THR = 0.07   # lakes/rivers
#     NDVI_THR = 0.18   # healthy green areas
#     NDBI_THR = 0.1    # built-up areas

#     # 1. NDWI → water (only where still unknown)
#     unknown = (final == 255) & inside
#     final[unknown & (ndwi >= NDWI_THR)] = 1

#     # 2. NDVI → vegetation
#     unknown = (final == 255) & inside
#     final[unknown & (ndvi >= NDVI_THR)] = 2

#     # 3. NDBI → built-up
#     unknown = (final == 255) & inside
#     final[unknown & (ndbi >= NDBI_THR)] = 3

#     # ─────────────────────────────────────
#     #  C. Nearest-neighbour fill for leftovers
#     # ─────────────────────────────────────
#     if np.any((final == 255) & inside):
#         print("🔁 Leftover unlabeled pixels – applying nearest neighbour fill...")
#         mask_unk = (final == 255)
#         _, (ni, nj) = distance_transform_edt(mask_unk, return_indices=True)
#         final[mask_unk] = final[ni[mask_unk], nj[mask_unk]]
#     else:
#         print("✅ No unknown pixels left inside city.")

#     # Enforce outside = 0 again (safety)
#     final[inside_mask == 0] = 0

#     # ── Save mask
#     meta.update({"count": 1, "dtype": "uint8"})
#     with rasterio.open(mask_out_fp, "w", **meta) as dst:
#         dst.write(final, 1)

#     print("\n✅ Year processed and saved:", mask_out_fp)
#     print("Class distribution for", year, ":")
#     print("  Outside(0):", int((final == 0).sum()))
#     print("  Water(1):  ", int((final == 1).sum()))
#     print("  Veg(2):    ", int((final == 2).sum()))
#     print("  Built(3):  ", int((final == 3).sum()))
#     print("Unique classes:", np.unique(final))

# print("\n🎉 All years processed successfully!")

import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
from scipy.ndimage import distance_transform_edt
import os
import re

# ── 1. Set root paths (Downloads folder)
ROOT = r"C:\Users\kiran\Downloads"
OSM_ROOT = ROOT  # OSM folders are directly in Downloads

print("\n🧾 Scanning available OSM extract folders...\n")
osm_dirs = [d for d in os.listdir(OSM_ROOT) if os.path.isdir(os.path.join(OSM_ROOT, d)) and 'southern-zone-' in d and 'free.shp' in d]
for d in osm_dirs:
    print(" -", d)

# ── 2. Load Bengaluru boundary region
print("\n🧭 Loading Bengaluru boundary file...\n")
boundary_fp = os.path.join(ROOT, "BENGALURU (1).geojson")
boundary = gpd.read_file(boundary_fp)
print("✔ Boundary loaded with geometry count:", len(boundary))


# ── 3. Function to decode year from folder names like southern-zone-220101-free.shp
def decode_year(folder_name):
    # Look for 6-digit date chunks like 220101, 230101 etc.
    nums = re.findall(r"(\d{2})(\d{2})(\d{2})", folder_name)
    if nums:
        y = nums[0][0]  # first two digits = year
        year = 2000 + int(y)  # all are 20xx in your case
        return year
    return None

# Mapping: year -> OSM folder path
year_to_osm = {}
for d in osm_dirs:
    year = decode_year(d)
    if year:
        year_to_osm[year] = os.path.join(OSM_ROOT, d)

# Manual mapping for your specific southern-zone folders
year_to_osm[2022] = os.path.join(ROOT, "southern-zone-220101-free.shp")
year_to_osm[2023] = os.path.join(ROOT, "southern-zone-230101-free.shp")
year_to_osm[2024] = os.path.join(ROOT, "southern-zone-230101-free.shp")  # Using 2023 data for 2024

print("\n🔎 OSM folders decoded to year mapping:\n")
for y, fp in sorted(year_to_osm.items()):
    print(f"  {y}: {fp}")

# ── 4. Output dir (create in Downloads folder)
out_mask_dir = os.path.join(ROOT, "test_masks")
os.makedirs(out_mask_dir, exist_ok=True)

print("\n🚀 Starting year-wise classification...\n")

# ── 5. Loop over years
for year in range(2022, 2025):
    print("\n──────────────")
    print(f"YEAR → {year}")
    print("──────────────")

    # Filenames for this year (all in Downloads folder)
    base_fp = os.path.join(ROOT, f"Landsat_{year}_RGBNIR.tif")
    ndvi_fp = os.path.join(ROOT, f"Landsat_{year}_NDVI.tif")
    ndwi_fp = os.path.join(ROOT, f"Landsat_{year}_NDWI.tif")
    ndbi_fp = os.path.join(ROOT, f"Landsat_{year}_NDBI.tif")
    mask_out_fp = os.path.join(out_mask_dir, f"{year}_mask.tif")
w
    # Check if required rasters exist
    if not os.path.exists(base_fp):
        print(f"❌ {base_fp} not found → skipping year {year}")
        continue
    if not (os.path.exists(ndvi_fp) and os.path.exists(ndwi_fp) and os.path.exists(ndbi_fp)):
        print(f"❌ NDVI/NDWI/NDBI missing for {year} → skipping")
        continue

    # Check OSM availability
    osm_year_dir = year_to_osm.get(year)
    if osm_year_dir:
        print("✅ OSM snapshot available for this year!")
        print("📍 OSM folder:", osm_year_dir)
    else:
        print("⚠️ No OSM for this year – will use spectral-only classification")

    # ── Read base raster to get canvas + CRS
    with rasterio.open(base_fp) as src:
        meta = src.meta.copy()
        shape_hw = (src.height, src.width)
        transform = src.transform
        crs = src.crs

    print("✔ Canvas size:", shape_hw)
    print("✔ CRS:", crs)

    # ── Reproject boundary to raster CRS
    boundary_pr = boundary.to_crs(crs)

    # ── Rasterize Bengaluru boundary (1 = inside, 0 = outside)
    inside_mask = features.rasterize(
        [(geom, 1) for geom in boundary_pr.geometry],
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        all_touched=True
    )

    # ── Initialize final labels: 255 = unknown
    final = np.full(shape_hw, 255, dtype="uint8")
    final[inside_mask == 0] = 0  # outside city
    inside = (inside_mask == 1)

    # ── Load spectral indices
    with rasterio.open(ndvi_fp) as src_ndvi:
        ndvi = src_ndvi.read(1)
    with rasterio.open(ndwi_fp) as src_ndwi:
        ndwi = src_ndwi.read(1)
    with rasterio.open(ndbi_fp) as src_ndbi:
        ndbi = src_ndbi.read(1)
    ndvi = np.nan_to_num(ndvi, nan=-1)
    ndwi = np.nan_to_num(ndwi, nan=0)
    ndbi = np.nan_to_num(ndbi, nan=-1)

    print("✔ NDVI min/max:", float(ndvi.min()), float(ndvi.max()))
    print("✔ NDWI min/max:", float(ndwi.min()), float(ndwi.max()))
    print("✔ NDBI min/max:", float(ndbi.min()), float(ndbi.max()))

    # ─────────────────────────────────────
    #  A. OSM-based labels (if available)
    # ─────────────────────────────────────
    if osm_year_dir:
        print("\n🗺 Applying OSM-based rasterization (built-up + water)...")

        # Built-up: roads, buildings, transport, railways, traffic, landuse, pois
        urban_layers = ["buildings", "roads", "transport", "railways", "traffic", "landuse", "pois"]
        for layer in urban_layers:
            layer_fp = os.path.join(osm_year_dir, f"gis_osm_{layer}_a_free_1.shp")
            if not os.path.exists(layer_fp):
                layer_fp = os.path.join(osm_year_dir, f"gis_osm_{layer}_free_1.shp")
            if os.path.exists(layer_fp):
                print(f"   → rasterizing urban layer: {layer}")
                gdf = gpd.read_file(layer_fp).to_crs(crs)
                if gdf.empty:
                    continue
                blr_clip = gdf.overlay(boundary_pr, how="intersection")
                if blr_clip.empty:
                    continue
                urban_r = features.rasterize(
                    [(geom, 1) for geom in blr_clip.geometry if geom is not None],
                    out_shape=shape_hw,
                    transform=transform,
                    fill=0,
                    all_touched=True
                )
                final[inside & (urban_r == 1)] = 3  # built-up

        # Water: water, waterways, water-like natural
        water_layers = ["water", "waterways", "natural"]
        for layer in water_layers:
            layer_fp = os.path.join(osm_year_dir, f"gis_osm_{layer}_a_free_1.shp")
            if not os.path.exists(layer_fp):
                layer_fp = os.path.join(osm_year_dir, f"gis_osm_{layer}_free_1.shp")
            if os.path.exists(layer_fp):
                print(f"   → rasterizing water layer: {layer}")
                gdf = gpd.read_file(layer_fp).to_crs(crs)
                if gdf.empty:
                    continue
                blr_clip = gdf.overlay(boundary_pr, how="intersection")
                if "fclass" in blr_clip.columns:
                    blr_clip = blr_clip[blr_clip["fclass"].isin(
                        ["water", "river", "reservoir", "lake", "pond", "basin"]
                    )]
                if blr_clip.empty:
                    continue
                water_r = features.rasterize(
                    [(geom, 1) for geom in blr_clip.geometry if geom is not None],
                    out_shape=shape_hw,
                    transform=transform,
                    fill=0,
                    all_touched=True
                )
                final[inside & (water_r == 1)] = 1  # water

    else:
        print("🛈 Skipping OSM step (no snapshot for this year).")

    # ─────────────────────────────────────
    #  B. Spectral rules
    # ─────────────────────────────────────
    print("\n🖌 Applying spectral NDWI/NDVI/NDBI rules...")

    # Thresholds
    NDWI_THR = 0.07   # lakes/rivers
    NDVI_THR = 0.18   # healthy green areas
    NDBI_THR = 0.1    # built-up areas

    # 1. NDWI → water (only where still unknown)
    unknown = (final == 255) & inside
    final[unknown & (ndwi >= NDWI_THR)] = 1

    # 2. NDVI → vegetation
    unknown = (final == 255) & inside
    final[unknown & (ndvi >= NDVI_THR)] = 2

    # 3. NDBI → built-up
    unknown = (final == 255) & inside
    final[unknown & (ndbi >= NDBI_THR)] = 3

    # ─────────────────────────────────────
    #  C. Nearest-neighbour fill for leftovers
    # ─────────────────────────────────────
    if np.any((final == 255) & inside):
        print("🔁 Leftover unlabeled pixels – applying nearest neighbour fill...")
        mask_unk = (final == 255)
        _, (ni, nj) = distance_transform_edt(mask_unk, return_indices=True)
        final[mask_unk] = final[ni[mask_unk], nj[mask_unk]]
    else:
        print("✅ No unknown pixels left inside city.")

    # Enforce outside = 0 again (safety)
    final[inside_mask == 0] = 0

    # ── Save mask
    meta.update({"count": 1, "dtype": "uint8"})
    with rasterio.open(mask_out_fp, "w", **meta) as dst:
        dst.write(final, 1)

    print("\n✅ Year processed and saved:", mask_out_fp)
    print("Class distribution for", year, ":")
    print("  Outside(0):", int((final == 0).sum()))
    print("  Water(1):  ", int((final == 1).sum()))
    print("  Veg(2):    ", int((final == 2).sum()))
    print("  Built(3):  ", int((final == 3).sum()))
    print("Unique classes:", np.unique(final))

print("\n🎉 All years processed successfully!")