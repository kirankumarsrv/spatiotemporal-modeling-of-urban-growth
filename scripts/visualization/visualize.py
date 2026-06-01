import rasterio
import matplotlib.pyplot as plt

with rasterio.open(r"C:\Users\kiran\Desktop\main el\Bengaluru_labeled_mask.tif") as src:
    img = src.read(1)

plt.figure()
plt.imshow(img, vmin=0, vmax=4)  # stretch 0→4 properly
plt.title("Built-up Mask Preview")
plt.show()
