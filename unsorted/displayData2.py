import numpy as np
import sys
import matplotlib.pyplot as plt


def comet(x,y=None, time=0.15):
    """
    Displays a comet plot

    by Sukhbinder
    date: 15 Feb 2021
    """
    x = np.asarray(x)
    plt.ion()
    plt.xlim(x.min(), x.max())
    if y is not None:
        y = np.asarray(y)
        plt.ylim(y.min(), y.max())
    else:
        plt.ylim(0, len(x))
    if y is not None:
        plot = plt.plot(x[0], y[0])[0]
    else:
        plot = plt.plot(x[0])[0]

    for i in range(len(x)+1):
        if y is not None:
            plot.set_data(x[0:i], y[0:i])
        else:
            plot.set_xdata(x[0:i])
        plt.draw()
        plt.pause(time)
    plt.ioff()

all_poses = np.fromfile(sys.argv[1], sep=' ').reshape((-1,12))
#detector_poses = all_poses[:,:3]

#displayed_poses = np.zeros((1,3))
#i = 0
#for pose in detector_poses:
    #if np.linalg.norm(pose) > 0.1:
    #if i > 0 and i < 1000000:
    #displayed_poses = np.append(displayed_poses,np.expand_dims(pose, axis=0), axis=0)
    #i = i +1

#plt.plot(displayed_poses[:,0], label="x")
#plt.plot(displayed_poses[:,1], label="y")
#plt.plot(displayed_poses[:,2], label="z")
#plt.legend()
#plt.show()
comet(all_poses[:,0],all_poses[:,1])
