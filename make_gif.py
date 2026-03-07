import glob
import os
from PIL import Image

def create_gif():
    # Find all synthesized images for the pix2pix run, sorted by their epoch number
    filenames = sorted(glob.glob("checkpoints/pix2pix/web/images/*synthesized_image.jpg"))
    
    if not filenames:
        print("No images found yet! Make sure you let it train past Epoch 5.")
        return

    frames = []
    for synth_path in filenames:
        # Load synthesized image
        synth_img = Image.open(synth_path)
        
        # Load corresponding ground truth (real) image
        real_path = synth_path.replace("synthesized_image", "real_image")
        
        if os.path.exists(real_path):
            real_img = Image.open(real_path)
            
            # Combine them side by side: Ground Truth (Left) | Prediction (Right)
            total_width = synth_img.width + real_img.width
            max_height = max(synth_img.height, real_img.height)
            
            combined_img = Image.new('RGB', (total_width, max_height))
            combined_img.paste(real_img, (0, 0))
            combined_img.paste(synth_img, (real_img.width, 0))
            
            frames.append(combined_img)
        else:
            frames.append(synth_img)

    if not frames:
        return

    # Save them as an animated GIF at 2 frames per second (duration in ms)
    frames[0].save(
        "training_progress_comparison.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=500,  # 500 ms = half a second per frame
        loop=0         # Loop forever
    )
    print(f"GIF saved! Compiled {len(frames)} frames into training_progress_comparison.gif")

if __name__ == "__main__":
    create_gif()
