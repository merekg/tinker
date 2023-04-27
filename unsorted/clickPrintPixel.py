import h5py
import matplotlib.pyplot as plt
import sys
import numpy as np
import png

def onclick(click):
    print("["
            + str(click.xdata)
            +","
            + str(click.ydata)
            +"]")

def sav(img):
    with open('/home/nview/Desktop/dpm004.png', 'wb') as f:
        writer = png.Writer(width=img.shape[1], height=img.shape[0], bitdepth=16,
                            greyscale=True)
        writer.write(f, img)

def displ(img):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(img)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return self.point

if __name__=="__main__":

    # get the volume
    with h5py.File(sys.argv[1], 'r') as stack_60_file:
        img_stack = stack_60_file['ITKImage/0/VoxelData'][:]

    avg_img = np.mean(img_stack, axis=0).astype(np.uint16)
    sav(avg_img)
 
    # displaying the image
    displ(avg_img)
