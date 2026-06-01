import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

with rasterio.open(r"C:\Users\kiran\Downloads\blr_yearly_masks\2018_mask.tif") as src:
# with rasterio.open(r"C:\Users\kiran\Downloads\2017_mask.tif") as src:

    mask = src.read(1)

print("Unique pixel labels:", np.unique(mask))

# Explicit color map for display
cmap = colors.ListedColormap([
    "#ffff00",  # 0 background
    "#1f78b4",  # 1 water  🌊
    "#33a02c",  # 2 veg    🌱
    "#ffffff",
    "#000000",  # 3 open   🌾
       # 4 built  🏙️
])
norm = colors.BoundaryNorm([0,1,2,3,4,5], cmap.N)

plt.figure()
plt.imshow(mask, cmap=cmap, norm=norm, interpolation="nearest")
plt.colorbar(ticks=[1,2,3,4], label="Classes")
plt.title("Built-up (4) + Land Cover Mask")
plt.show()
