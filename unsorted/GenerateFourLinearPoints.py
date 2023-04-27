import numpy as np
import sys
import matplotlib.pyplot as plt

# constants
DISTORTION_VALUE = .1
N = 1

if __name__ == "__main__":
    seeds = np.fromfile(sys.argv[1], sep=' ').reshape((-1,4,3))
    for seed in seeds:
        for i in range(N):
            delta = DISTORTION_VALUE * (np.random.rand(seed.shape[0], seed.shape[1]) - .5)
            delta[:,0] = 0
            print(*list(seed + delta))

