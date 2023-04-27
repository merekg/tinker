import numpy as np
import sys
from PIL import Image
import matplotlib.pyplot as plt

dpm_image = Image.open(sys.argv[1])
dpm = np.array(dpm_image)
bad_pixels = np.argwhere(dpm>0)
for p in bad_pixels:
    print("[" + str(p[1]) + "," + str(p[0]) + "]")
