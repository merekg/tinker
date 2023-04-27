import numpy as np
import sys

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
print(*np.mean(arr,axis=0))
