import richdem as rd
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# File names and threshold
input_file = 'Lidar.tif'
output_file = 'Lidar_test.tif'
min_sinkhole_depth = 0.5  # in the same units as your DEM

# --------------------
# Step 1: Read DEM and Prepare Data
# --------------------
with rasterio.open(input_file) as src:
    # Read data as a masked array.
    dem_ma = src.read(1, masked=True)
    profile = src.profile
    # Convert data to float32 and assign NaN to masked locations.
    dem = dem_ma.astype('float32').data
    dem[dem_ma.mask] = np.nan

# If negative elevations are not expected, set them to NaN.
dem[dem < 0] = np.nan

# --------------------
# Step 2: Sinkhole Detection
# --------------------
# Convert DEM to a RichDEM array.
rdem = rd.rdarray(dem, no_data=np.nan)
rdem.metadata = {'name': 'Original DEM'}

# Fill depressions using the Priority-Flood algorithm.
filled_dem = rd.FillDepressions(rdem, in_place=False)

# Compute the depression depth (filled DEM minus original).
diff = filled_dem - rdem

# Create a binary mask where the depression exceeds the threshold.
sinkhole_mask = (diff > min_sinkhole_depth).astype(np.uint8)

# --------------------
# Step 3: Save the Sinkhole Mask as a GeoTIFF
# --------------------
# Update the profile for single-band uint8 output.
profile.update({
    'dtype': 'uint8',
    'count': 1,
    'nodata': 0
})

with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(sinkhole_mask, 1)

print(f"Sinkhole detection completed. Output saved to {output_file}")

# --------------------
# Step 4: Visualization: Overlay Sinkholes in Orange on the DEM
# --------------------
# Read the original DEM for visualization.
with rasterio.open(input_file) as src:
    dem_viz = src.read(1)

# Read the sinkhole mask.
with rasterio.open(output_file) as mask_src:
    sink_mask_viz = mask_src.read(1)

# Create a plot.
plt.figure(figsize=(10, 10))
plt.imshow(dem_viz, cmap='gray')
# Mask non-sinkhole areas so only sinkhole pixels are overlaid.
plt.imshow(np.ma.masked_where(sink_mask_viz == 0, sink_mask_viz), cmap='Oranges', alpha=0.6)
plt.title('LIDAR DEM with Sinkholes Highlighted in Orange')
plt.axis('off')
plt.colorbar(label='Elevation')
plt.show()
