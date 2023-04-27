import numpy as np
import sys
import os

if __name__ == "__main__":


    assert len(sys.argv) == 3
    assert os.path.isfile(sys.argv[1])

    inpath = sys.argv[1]
    infile = open(inpath, 'r')
    lines = infile.readlines()

    for line in lines:
        count += 1
        print("Line{}: {}".format(count, line.strip()))
