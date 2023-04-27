import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np

PROBE_ARRAY = [[-14.163,93.847,-853.057],
               [131.710,77.142,-917.967],
               [-77.521,101.341,-824.547],
               [-108.621,62.544,-818.771],
               [-100.877,147.740,-805.548],
               [-145.872,109.769,-794.029]]

PATIENT_ARRAY = [[96.047,123.918,-951.638],
                 [105.648,40.348,-939.656],
                 [132.820,60.804,-891.345],
                 [127.698,105.060,-898.554]]
DIP = [[96.081],[123.943],[-951.744]]
TIP = [131.710,77.142,-917.967]
Q = [0.3272609,-0.6706059,0.5488676,-0.3767392]
ORIGIN = [[55],[-38],[12.138]]

r = np.array(R.from_quat([Q[1], Q[2], Q[3], Q[0]]).as_matrix())
originTransformed = np.matmul(r, ORIGIN) + DIP
print(originTransformed)
print(TIP)
print(np.linalg.norm(np.array(originTransformed).flatten() - np.array(TIP).flatten()))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for mark in PROBE_ARRAY:
    ax.scatter(mark[0], mark[1], mark[2], marker='o', color='y')
for mark in PATIENT_ARRAY:
    ax.scatter(mark[0], mark[1], mark[2], marker='o', color='g')
ax.scatter(originTransformed[0],originTransformed[1], originTransformed[2], marker='x', color='r')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
