import sys
import numpy as np

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

M1 = np.fromfile(sys.argv[1],sep=' ').reshape((4,4))
M2 = np.fromfile(sys.argv[2],sep=' ').reshape((4,4))

print(M1 @ invert_matrix(M1))
print(M2 @ invert_matrix(M2))
print(M1 @ invert_matrix(M2))
print(M2 @ invert_matrix(M1))
