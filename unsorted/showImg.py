import numpy as np
import sys
import matplotlib.pyplot as plt
import h5py

# constants
VOXELDATA_PATH = "ITKImage/0/VoxelData"

def getImageFromH5(path):
    with h5py.File(path, 'r') as imgFile:
        return imgFile[VOXELDATA_PATH][:]

def generateDeadPixelMap(path, map_x, map_y):
    dead_pix_list = []
    for line in open(path, "r"):
        line = line.replace("[","").replace("]","")
        x,y = line.split(",")
        dead_pix_list.append([int(x), int(y)])
    dead_pix_list = np.array(dead_pix_list)
    dead_pix_map = np.zeros((map_x,map_y))
    for coord in dead_pix_list:
        x_i = coord[0]
        y_i = coord[1]
        dead_pix_map[x_i,y_i] = 1
    return dead_pix_map

def displayImage(img):
    plt.imshow(img)
    plt.show()

def main():
    for path in sys.argv[1:]:
        dpm = generateDeadPixelMap(path, 976, 976)
        displayImage(dpm)

if __name__ == "__main__":
    main()
