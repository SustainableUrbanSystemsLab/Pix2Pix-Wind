import os
import imageio.v2 as imageio
import re

# Your folder containing PNGs
folder = r"C:\Users\zeyuj\Downloads\epoch_visuals"

# Output GIF
output_gif = r"C:\Users\zeyuj\Downloads\training.gif"

# Find PNG files
png_files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]

# Sort by epoch number so frames are correct
png_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

frames = []

for file in png_files:
    path = os.path.join(folder, file)
    img = imageio.imread(path)
    frames.append(img)

# Create GIF
imageio.mimsave(output_gif, frames, duration=0.08)

print("GIF saved to:", output_gif)