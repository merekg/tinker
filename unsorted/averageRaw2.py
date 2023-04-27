import numpy as np
import sys

a = np.fromfile(sys.argv[1],sep=' ').reshape((-1,18))
print(*np.mean(a, axis=0))
print(np.std(a,axis=0))
