import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import truncnorm
import math

PI = math.pi

def start_branch():
    r = np.random.rand(1)
    return r > .97

def get_path_length(mean, sd=50):
    low = mean - PI
    upp = mean + PI
    return int(truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs())

def get_next_direction(mean, sd=PI/12):
    low = mean - PI
    upp = mean + PI
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs()

def get_next_pose(pose,direction):
    next_direction = get_next_direction(direction)
    x = pose[0] + math.cos(direction)
    y = pose[1] + math.sin(direction)

    return np.array([[x,y]]), next_direction

def get_path(pose, direction, length):
    direction = get_next_direction(direction, PI/4)
    length = get_path_length(length)
    for i in range(length):
        next_pose, next_direction = get_next_pose(pose[-1],direction)
        direction = next_direction
        pose = np.append(pose, next_pose, axis=0)
        if start_branch() and length > 15:
            pose = np.append(pose, get_path(next_pose, next_direction, length//2), axis=0)
    return pose

pose = get_path(np.array([[0.0,0.0]]), 0, 100)
plt.plot(pose[:,0],pose[:,1])
plt.show()
