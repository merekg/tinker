import numpy as np
import sys


def get_average(camera_path):
    f = open(camera_path, 'r')
    average_marker_centers = np.zeros_like(np.array(f.readlines()[0].strip()[:-1].split(",")).astype(np.float))
    nav_file = open(camera_path,'r')
    n_samples = 0
    for line in nav_file:
        if "nan" in line:
            continue
        n_samples += 1
        average_marker_centers = average_marker_centers + np.array(line.replace(' ','').strip()[:-1].split(",")).astype(np.float)

    average_marker_centers /= n_samples
    print(*average_marker_centers)

get_average(sys.argv[1])
