import numpy as np
import sys

def point_from_matrix(M):
    p = np.array([1,0,0,1])
    return (M @ p)[:-1]

def main():
    pose1 = point_from_matrix(np.fromfile(sys.argv[1], sep=' ').reshape(4,4))
    pose2 = point_from_matrix(np.fromfile(sys.argv[2], sep=' ').reshape(4,4))
    print(np.linalg.norm(pose2 - pose1))

if __name__ == "__main__":
    main()
