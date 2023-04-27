import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R

INTRINSIC_ORIENTATION = np.array([1,0,0])
PROBE = np.array([
[-159.44,-1.16,-0.71],
[-229.29,-1.16,-0.71],
[-255.45,-43.83,-0.71],
[-263.07,42.27,-0.71],
[-304.47,-1.16,-0.71]])

def display_cloud(source):
    source = np.array(source)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim(-300,300)
    ax.set_ylim(-300,300)
    ax.set_zlim(-1400,-800)
    xs, ys, zs = source[:,0], source[:,1], source[:,2]
    ax.scatter(xs, ys, zs, marker='o')

    plt.show()

def rotation_between(v1,v2):
    a, b = (v1 / np.linalg.norm(v1)).reshape(3), (v2 / np.linalg.norm(v2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

f = open(sys.argv[1],'r')

arr = []
for line in f:
    current_line_array = []
    for el in line.split(","):
        try:
            current_line_array.append(float(el))
        except:
            pass
    if len(current_line_array) > 0:
        arr.append(current_line_array)
arr = np.array(arr)
ave = np.mean(arr,axis=0)
q = ave[3:7]
t = ave[7:10]

r = R.from_quat(np.array([q[1],q[2],q[3],q[0]])).as_matrix()
transformed = np.array([ r@m + t for m in PROBE])
display_cloud(transformed)
print(transformed)
