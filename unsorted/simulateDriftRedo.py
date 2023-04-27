import numpy as np
import sys

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

old_cal = np.array([[-1.089171798315459461e-02,-1.543011868387725305e-02,-9.998216250495768342e-01,-7.575906501277559357e+01],
                   [-3.974883885690979612e-01,9.175545456460173499e-01,-9.830397605060990293e-03,-6.022656425321027740e+01],
                   [9.175425611011824856e-01,3.973104166791164849e-01,-1.612703835642247119e-02,-2.731187103873490969e+02],
                   [0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])

nav_points = np.fromfile(sys.argv[1],sep=' ').reshape((-1,3))
viewer_points = np.fromfile(sys.argv[2],sep=' ').reshape((-1,3))
cal = np.fromfile(sys.argv[3],sep=' ').reshape((4,4))

for n in nav_points:
    t = (cal @ (invert_matrix(old_cal) @ np.append(n,1)))[:3]
    print(*t)
