import numpy as np
import sys
import os

def openMatrix(filepath):
    return np.fromfile(filepath, dtype=float, sep=' ').reshape((4,4))

def openPoint(filepath):
    a = np.fromfile(filepath, dtype=float, sep=' ')
    return np.append(a, 1)


def main():
    assert len(sys.argv) is 6
    p1 = openPoint(sys.argv[1])
    p2 = openPoint(sys.argv[2])
    tc = openMatrix(sys.argv[3])
    td1 = openMatrix(sys.argv[4])
    td2 = openMatrix(sys.argv[5])

    tx = np.matmul(tc, np.matmul(td1, np.matmul(np.linalg.inv(td2), np.linalg.inv(tc))))
    p2_calc = np.matmul(tx, p1)
    print("p2: " + str(p2))
    print("p2_calc: " + str(p2_calc))

if __name__ == "__main__":
    main()
