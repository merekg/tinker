import numpy as np
import sys
import matplotlib.pyplot as plt

all_poses = np.fromfile(sys.argv[1], sep=' ').reshape((-1,12))
detector_poses = all_poses[:,:3]

displayed_poses = np.zeros((1,3))
#i = 0
#for pose in detector_poses:
    #if np.linalg.norm(pose) > 0.1:
    #if i > 0 and i < 1000000:
    #displayed_poses = np.append(displayed_poses,np.expand_dims(pose, axis=0), axis=0)
    #i = i +1

plt.plot(all_poses[:,0], label="x")
plt.plot(all_poses[:,1], label="y")
plt.plot(all_poses[:,2], label="z")
plt.legend()
plt.show()
