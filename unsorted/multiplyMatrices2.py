import sys
import numpy as np

A = np .fromfile(sys.argv[1], sep=' ').reshape((4,4))
B = np.fromfile(sys.argv[2], sep=' ').reshape((4,4))

# format the print how I want
for row in np.matmul(A,B):
    print(*row)
