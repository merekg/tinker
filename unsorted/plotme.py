import matplotlib.pyplot as plt
import numpy as np
import sys

camera_points = np.fromfile(sys.argv[1], sep=' ').reshape((-1,3))
volume_points = np.fromfile(sys.argv[2], sep=' ').reshape((-1,3))

ax = plt.figure().add_subplot(projection='3d')

# Make the grid
x = camera_points[:,0]
y = camera_points[:,1]
z = camera_points[:,2]

print(x)
# Make the direction data for the arrows
u = camera_points[:,0] - volume_points[:,0]
v = camera_points[:,1] - volume_points[:,1]
w = camera_points[:,2] - volume_points[:,2]

ax.quiver(x, y, z, u, v, w, length=10, normalize=True)

plt.show()
