from PIL import Image
import sys
import os

for path in sys.argv[1:]:
    img = Image.open(os.path.abspath(path))
    img = img.crop((11,177,748,914))
    img.save(os.path.abspath(path))
