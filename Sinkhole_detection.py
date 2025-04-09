import numpy as np
import richdem as rd
import rasterio
from earthpy.spatial import hillshade

def process_tif(input_file, output_file, threshold=0.5, azimuth=315, altitude=30):
    # Open the input DEM and read the elevation data.
    with rasterio.open(input_file) as src:
        elevation = src.read(1, masked=True).filled(fill_value=0)
        profile = src.profile

    # Ensure there are no negative elevations.
    elevation[elevation < 0] = 0

    # Create a richdem array and fill depressions.
    rd_dem = rd.rdarray(elevation, no_data=0)
    filled = rd.FillDepressions(rd_dem)
    diff = np.array(filled - rd_dem)  # Sinkhole gradient: difference between filled and original

    # Compute hillshade for visual reference.
    hs = hillshade(elevation, azimuth=azimuth, altitude=altitude)
    hs = hs.astype(np.uint8)

    # Create an RGB image with hillshade as the background.
    img_rgb = np.stack([hs, hs, hs], axis=0)

    # Create a mask for the sinkhole regions where the gradient exceeds the threshold.
    sinkhole_mask = diff >= threshold

    # Overlay sinkhole pixels as red (set red channel to 255, green and blue to 0).
    img_rgb[0, sinkhole_mask] = 255  # Red channel
    img_rgb[1, sinkhole_mask] = 0    # Green channel
    img_rgb[2, sinkhole_mask] = 0    # Blue channel

    # Update the profile for a 3-band (RGB) output with type uint8.
    profile.update(dtype=rasterio.uint8, count=3)

    # Write the output TIFF file.
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(img_rgb)

    print(f"Output written to {output_file}")

# Specify your file names here:
input_filename = 'your_input_file.tif'  # Replace with the name or path to your DEM file.
output_filename = 'your_output_file.tif'  # Replace with the desired output file name.

# Run the process
process_tif(input_filename, output_filename, threshold=0.5)
