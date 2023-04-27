import sys
import numpy as np

arr =  np.fromfile(sys.argv[1], dtype=str,sep=' ')
print(arr.shape)

for line in arr:
    print(line)
    print("[" + line[1] + "," + line[0] + "]")
