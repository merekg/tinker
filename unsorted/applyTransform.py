import sys
import os
import numpy as np

campath = sys.argv[1]
matrixPath = sys.argv[2]
volpath = sys.argv[3]
outputPath = sys.argv[4]

cameraPoints = np.fromfile(campath,dtype=float, sep=',')
T = np.fromfile(matrixPath,dtype=float, sep=',')
volPoints = np.fromfile(volpath,dtype=float, sep=',')

for cameraPoint in cameraPoints:
    trans = np.matmul(T, np.append(cameraPoint,1).T)
    print(np.linalg.norm(trans - volPoints))
