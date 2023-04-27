import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import sys
import os
from matplotlib.widgets import Slider, Button

SPACE_TO_PIXELS = 10
  
def update(val):
    fig.canvas.draw_idle()

def main():
    global translucentLayer
    global rawMap
    global fig
    global ax
    global radiusSlider

    rawMap = np.asarray(ImageOps.grayscale(Image.open(os.path.abspath(sys.argv[1]))))
    translucentLayer = np.zeros([rawMap.shape[0], rawMap.shape[1]])
    opaqueLayer = np.zeros([rawMap.shape[0], rawMap.shape[1]]) 

    fig = plt.figure()
    ax = fig.add_subplot(111)

    axamp = plt.axes([0.1, 0.25, 0.0225, 0.63])
    radiusSlider = Slider(
        ax=axamp,
        label="Vision (m)",
        valmin=0,
        valmax=100,
        valinit=10,
        orientation="vertical"
    )

    ax.imshow(rawMap * np.mean([opaqueLayer, translucentLayer], axis=0), cmap='gray')
    for i in range(0,1):
        cid = fig.canvas.mpl_connect('button_press_event', onClick)
    plt.show()

def getPoseAndRadius():
    pose = [int(i) * SPACE_TO_PIXELS for i in input("Pose:").split()]
    radius = int(input("Radius:")) * SPACE_TO_PIXELS
    return pose, radius

def updateLayers(pose, radius, translucentLayer):
    mask = create_circular_mask(translucentLayer.shape[0], translucentLayer.shape[1], pose, radius)
    translucentLayer = np.where(mask, mask, translucentLayer)
    return translucentLayer, mask

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def onClick(event):
    global ix, iy
    global translucentLayer, rawMap, ax, radiusSlider
    ix, iy = event.xdata, event.ydata
    if( ix < 1):
        return
    print("x=" + str(ix) + ", y=" + str(iy))

    for i in range(0,1):
        cid = fig.canvas.mpl_connect('button_press_event', onClick)
    translucentLayer, opaqueLayer = updateLayers([ix,iy], radiusSlider.val * SPACE_TO_PIXELS, translucentLayer)
    ax.imshow(rawMap * np.mean([opaqueLayer, translucentLayer], axis=0), cmap='gray')
    fig.canvas.draw()
    fig.canvas.flush_events()
    
if __name__ == "__main__":
    main()
