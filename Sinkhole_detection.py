import richdem as rd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import measure

# --- Parameters ---
input_file = 'Lidarsample.tif'
output_file = 'Lidarsampleoutput.tif'

# Detection & filtering parameters
min_sinkhole_depth = 0.5   # Minimum depression depth (DEM units) to consider a sinkhole
min_area = 45              # Only consider regions with at least 45 pixels
min_circularity = 0.5      # Only consider regions with circularity >= 0.5

# Colors for gradient: shallow (edge) is yellow-orange, deep (center) is black.
shallow_color = np.array([255, 165, 0], dtype=np.float32)  # Yellow-orange
deep_color = np.array([0, 0, 0], dtype=np.float32)           # Black

# --------------------
# Step 1: Read DEM and Prepare Data
# --------------------
with rasterio.open(input_file) as src:
    dem_ma = src.read(1, masked=True)  # Read as a masked array.
    profile = src.profile.copy()
    transform = src.transform

# Convert to float32 and assign NaN to masked pixels.
dem = dem_ma.astype(np.float32).data
dem[dem_ma.mask] = np.nan

# (Optional) Set negative elevations to NaN if appropriate.
dem[dem < 0] = np.nan

# --------------------
# Step 2: Fill Depressions with RichDEM and Compute Depression Depth
# --------------------
rdem = rd.rdarray(dem, no_data=np.nan)
rdem.metadata = {'name': 'Original DEM'}
filled_dem = rd.FillDepressions(rdem, in_place=False)
diff = filled_dem - rdem  # Depression depth

# Apply a threshold: only consider depressions deeper than min_sinkhole_depth.
depression = diff.copy()
depression[depression < min_sinkhole_depth] = 0

# --------------------
# Step 3: Label Depressions and Filter by Circularity
# --------------------
# Create binary mask (1 where depression > 0)
binary = (depression > 0).astype(np.uint8)
labeled, num = ndi.label(binary)
regions = measure.regionprops(labeled, intensity_image=depression)

# Initialize an empty overlay for the color gradient (3 channels for RGB)
overlay = np.zeros((3, dem.shape[0], dem.shape[1]), dtype=np.uint8)
region_count = 0

for region in regions:
    if region.area < min_area:
        continue
    if region.perimeter == 0:
        continue
    circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
    if circularity < min_circularity:
        continue

    # Debug: print stats for qualifying regions
    print(f"Region {region.label}: area={region.area}, circularity={circularity:.2f}")

    # Create mask for the current region.
    region_mask = (labeled == region.label)
    region_depth = depression[region_mask]
    
    # Normalize depression in this region: shallow edge -> 0; deepest -> 1.
    if region_depth.ptp() < 1e-6:
        normalized = np.zeros_like(region_depth)
    else:
        normalized = (region_depth - region_depth.min()) / (region_depth.ptp() + 1e-6)
    normalized = np.clip(normalized, 0, 1)
    
    # Compute pixel color by interpolating from shallow_color (when normalized=0) to deep_color (when normalized=1).
    # Since deep_color is [0,0,0] (black), this becomes:
    # pixel_color = shallow_color * (1 - normalized)
    color_array = (shallow_color[None, :] * (1 - normalized[:, None])).astype(np.uint8)
    
    # Assign the computed colors to the overlay for pixels in this region.
    coords = np.column_stack(np.where(region_mask))
    for (y, x), color in zip(coords, color_array):
        overlay[:, y, x] = color
    region_count += 1

print(f"Number of regions after filtering: {region_count}")

# --------------------
# Step 4: Composite with Grayscale Background
# --------------------
# Create a grayscale background from the original DEM.
gray = np.interp(dem, (np.nanmin(dem), np.nanmax(dem)), (0, 255)).astype(np.uint8)
background = np.stack([gray] * 3, axis=0)
# Where the overlay has nonzero color, use it; otherwise, fall back to the grayscale background.
final = np.where(overlay.sum(axis=0) > 0, overlay, background)

# --------------------
# Step 5: Save the Final Composite as a GeoTIFF
# --------------------
profile.update({
    'count': 3,      # 3 channels for RGB
    'dtype': 'uint8'
})
with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(final)

print(f"Sinkhole detection with circularity filtering and gradient overlay completed. Output saved to {output_file}")

# --------------------
# Step 6: Visualization
# --------------------
plt.figure(figsize=(10, 10))
plt.imshow(np.moveaxis(final, 0, -1))
plt.title("Detected Sinkholes with Circularity Filter and Gradient")
plt.axis("off")
plt.show()
