import numpy as np
import sys

m1 = np.fromfile(sys.argv[1],sep=' ').reshape((4,4))
print(m1)
m2 = np.fromfile(sys.argv[2],sep=' ').reshape((4,4))
print(m2)

print( m1 @ m2)
