import numpy as np
import sys
import os
from PIL import Image

INTENSITY_CHARACTERS = [' ', '`', '.', '\'', '\"', '*', 'O', '0', '#', '@']
INTENSITY_BIN_COUNT = len(INTENSITY_CHARACTERS)

def bin_img(img, new_shape):
    print(img.shape)
    print(new_shape)
    shape = (new_shape[0], img.shape[0] // new_shape[0],
             new_shape[1], img.shape[1] // new_shape[1])
    return img.reshape(shape).mean(-1).mean(1)

def img_path_to_array(path):
    return np.asarray(Image.open(path).convert('L'))

def img_to_ascii_art(img):
    M = np.max(img)
    ascii_art = ""
    for row in img:
        for pix in row:
            ascii_art += INTENSITY_CHARACTERS[ int(np.floor(pix * INTENSITY_BIN_COUNT / M)) - 1]
        ascii_art += '\n'
    return ascii_art

def greatest_common_factor_2(n,m):
    gcf = -1

    for power in range(0,max(n,m)):
        cf = 2**power
        if n%cf is 0 and m%cf is 0:
            gcf = cf
    return gcf

if __name__ == "__main__":

    # get an image from the user
    image_path = os.path.abspath(sys.argv[1])
    img = img_path_to_array(image_path)

    # bin the image as much as possible
    shape = np.divide(img.shape, greatest_common_factor_2(*img.shape)).astype(int)
    binned_img = bin_img(img, shape)

    # Convert the image to ascii
    ascii_img = img_to_ascii_art(binned_img)

    # print the ascii art
    print(ascii_img)
