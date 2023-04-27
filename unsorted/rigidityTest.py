#! /usr/bin/env python3
import numpy as np
from scipy import ndimage
import nrrd
import sys
import skg
import argparse
import os
import matplotlib.pyplot as plt

N_MARKERS = 6
MAX_ERROR_MM = 1.0
MIN_CAMERA_SAMPLES = 100

# These constants are the distances between any array and all other arrays.
# comparing the clouds to these signatures will tell us which marker is which
# so that we can match them up consistently.
FIXTURE_ARRAY = np.array([[0,90,0],[250,94.4,0],[130,115,0],[12,189,0],[140,200,0],[220,183.5,0]])
SIGNATURE_ERROR = 10

def apply_rigid_transform(a, T):
    a_0 = np.append(a, 0)
    a_0_t = np.matmul(T, a_0)
    return a_0_t[:-1]

def display_clouds(source, target):
    source = np.array(source)
    target = np.array(target)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-300,300)
    ax.set_ylim(-300,300)
    ax.set_zlim(-300,300)

    xs, ys, zs = source[:,0], source[:,1], source[:,2]
    ax.scatter(xs, ys, zs, marker='o')

    xt, yt, zt = target[:,0], target[:,1], target[:,2]
    ax.scatter(xt, yt, zt, marker='x')
    ax.legend(["volume","navigation"])

    plt.show()

def to_transform_matrix(rotationMatrix, translationVector):

    # Start the transformation matrix as the identity matrix
    T = np.identity(4)

    # Fill in the rotation and translation parts of the transformation matrix
    T[0:3,0:3] = rotationMatrix
    T[0:3,3] = translationVector

    return T

def invert_matrix(T):
    r = T[:3, :3]
    t = T[:3, 3]
    ri = np.linalg.inv(r)
    ti = - np.matmul(ri, t)
    Ti = np.zeros_like(T)
    Ti[:3,:3] = ri
    Ti[:3,3] = ti
    Ti[3,3] = 1
    return Ti

def compute_rigid_transform(camera_points, volume_points):

    # Verify that: the shape of camera_points and volume_points match, each have 3 rows
    assert camera_points.shape[0] == 3
    assert volume_points.shape[0] == 3

    # Compute the centroid of each matrix
    Ac = np.mean(camera_points, axis=1)[:,None]
    Bc = np.mean(volume_points, axis=1)[:,None]

    # Compute covariance of centered matricies and find rotation using SVD
    H = np.matmul(camera_points - Ac, (volume_points - Bc).T)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # Correct R for the special reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.matmul(Vt.T, U.T)

    # Compute translation from centroids and rotation matrix
    t = np.matmul(-R, Ac) + Bc

    return R, t

def check_transformation(A, B, T):

    # Transform the points A by the transformation T
    TA = np.matmul(T, np.vstack((A, np.ones(len(A[0])))))[:-1]

    # Find the difference with B and compute euclidian the error
    error = B - TA
    linearError = np.linalg.norm(error, axis=0)

    b_disp = B
    ta_disp = TA
    display_clouds(b_disp.T, ta_disp.T)
    # Return the error values
    return linearError


def main():

    all_volume_centers = np.fromfile(sys.argv[1], sep=' ').reshape((-1,6,3))
    transformed_volume_centers = all_volume_centers
    array_home_pose = all_volume_centers[0]

    for center in all_volume_centers:
        # Compute rigid transform
        rotation, translation = compute_rigid_transform(array_home_pose.T, center.T)
        registration_matrix = to_transform_matrix(rotation, translation.flatten())

        # Perform the check to see how well the points register
        error = check_transformation(array_home_pose.T, center.T, registration_matrix)
        print("Registration error:",np.max(np.abs(error)))

if __name__ == "__main__":
    main()
