import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

vectors = np.fromfile(sys.argv[1],sep=' ').reshape((-1,3))
directions = np.fromfile(sys.argv[2],sep=' ').reshape((-1,4))
for d,v in zip(directions,vectors):
    r = R.from_quat( np.array([d[1], d[2],d[3],d[0]])).as_matrix()
    result = np.matmul(r.T,v)
    print(result[0],",",result[1],",",result[2])
