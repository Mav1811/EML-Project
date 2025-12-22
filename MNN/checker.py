import struct
import numpy as np
from PIL import Image
import os

# ---- paths ----
mnist_images = "/home/orangepi/Documents/data/MNIST/raw/train-images-idx3-ubyte"
output_dir = "calib_RGB"

NUM_IMAGES = 200
ROWS = 28
COLS = 28

os.makedirs(output_dir, exist_ok=True)

with open(mnist_images, "rb") as f:
    # Read MNIST header (big-endian)
    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))

    print(f"Magic: {magic}, Images: {num}, Size: {rows}x{cols}")

    for i in range(NUM_IMAGES):
        # Read one image
        img_bytes = f.read(ROWS * COLS)

        # Convert to numpy array
        img = np.frombuffer(img_bytes, dtype=np.uint8)
        img = img.reshape((ROWS, COLS))

        # Save as grayscale JPG
        image = Image.fromarray(img, mode="L")
        image.save(f"{output_dir}/{i}.jpg", format="JPEG")

print("Saved 200 MNIST images as JPG in ./calib/")
