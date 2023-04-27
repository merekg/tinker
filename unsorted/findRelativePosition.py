import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.spatial.transform import Rotation as R
probe_markers = np.array([[196.33675107,-39.51839579,-1039.40246906],
 [  186.6460043,    23.18965304,-1068.60542333],
 [  164.01566807,   28.17188437,-1112.96891905],
 [  201.29890147,   72.3482968 ,-1048.70633668],
 [  176.21579182,   90.68272551,-1100.03674905]])

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


def to_rotation_matrix(qx,qy,qz,qw):
    return R.from_quat([qx,qy,qz,qw]).as_matrix()

def to_transform_matrix(R,t):
    T = np.identity(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def invert_matrix(T):
    r = T[:3, :3]
    t = T[:3, 3]
    ri = np.linalg.inv(r)
    ti = - np.matmul(ri, t)
    Ti = np.zeros_like(T)
    Ti[:3,:3] = ri
    Ti[:3,3] = ti
    Ti[3,3] = 1
    return Ti

def main():
    detector_file = np.fromfile(sys.argv[1],sep=' ')

    detector_r = to_rotation_matrix(detector_file[4],detector_file[5],detector_file[6],detector_file[3])
    detector_t = np.array([detector_file[7],detector_file[8],detector_file[9]])
    T = to_transform_matrix(detector_r,detector_t)

    tx = np.array([invert_matrix(T) @ np.append(m,[1]) for m in probe_markers])[:,:3]
    display_cloud(tx)
    print(tx)

if __name__ == "__main__":
    main()
