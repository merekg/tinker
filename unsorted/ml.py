import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

IMAGE_SHAPE = [28,28]

def show_image(path):
    img = mpimg.imread(path)
    imgplot = plt.imshow(img)
    plt.show()

def open_mnist(path):
    raw = np.genfromtxt(path,delimiter=',')
    return raw[:,0], raw[:,1:]

def ReLU(Z):
    return np.max(0,Z)

def d_ReLU(Z):
    return Z > 0

def main():
    labels, d = open_mnist(sys.argv[1])
    print(d[0].shape)

if __name__ == "__main__":
    main()
