import sys
from PIL import Image
import random

# filepaths
fp_in = sys.argv[1:-1]
fp_out = sys.argv[-1]

imgs = (Image.open(f) for f in random.shuffle(fp_in))
img = next(imgs)  # extract first image from iterator
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)
