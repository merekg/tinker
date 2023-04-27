import numpy as np
import sys
from scipy.spatial.transform import Rotation as R

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
    data = np.fromfile(sys.argv[1],sep=' ').reshape((-1,14))

    for line in data:
        patient_reference_R = to_rotation_matrix(line[1],line[2],line[3],line[0])
        system_R = to_rotation_matrix(line[8],line[9],line[10],line[7])
        pT = to_transform_matrix(patient_reference_R, [line[4],line[5],line[6]])
        sT = to_transform_matrix(system_R, [line[11],line[12],line[13]])
        print(*(invert_matrix(sT) @ [line[4],line[5],line[6],1])[:3])

if __name__ == "__main__":
    main()
