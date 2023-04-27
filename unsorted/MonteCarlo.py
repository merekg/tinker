import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

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
    #display_clouds(b_disp.T, ta_disp.T)
    # Return the error values
    return linearError

def get_marker_variations(markers):

    variations = []

    d = np.array([[0.3,0,0],
                  [0,0.3,0],
                  [0,0,0.3],
                  [-0.3,0,0],
                  [0,-0.3,0],
                  [0,0,-0.3]])
    
    for i in range(markers.shape[0]):
        for delta in d:
            markers_copy = markers
            markers_copy[i] = markers_copy[i] + np.array(delta)
            print(", ".join(markers.flatten().astype(str).tolist()))
            markers_copy[i] = markers_copy[i] - np.array(delta)

    return variations

def distance_between_rotation_matrices(P,Q):
    R = P @ Q.T
    theta = np.arccos((R.trace() - 1)/2)
    return np.degrees(theta)

def to_transform_matrix(rotationMatrix, translationVector):

    # Start the transformation matrix as the identity matrix
    T = np.identity(4)

    # Fill in the rotation and translation parts of the transformation matrix
    T[0:3,0:3] = rotationMatrix
    T[0:3,3] = translationVector

    return T

def print_colinearity_metrics(cloud):
    # Compute singular values for camera_points
    _, singular_values, _ = np.linalg.svd(cloud - np.mean(cloud, axis=1)[:,None])
    major_sv = np.max(singular_values)
    minor_sv_sum = np.sum(singular_values) - major_sv
    print(major_sv, minor_sv_sum)

def display_clouds(source):
    source = np.array(source)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-300,300)
    ax.set_ylim(-300,300)
    ax.set_zlim(-300,300)

    xs, ys, zs = source[:,0], source[:,1], source[:,2]
    ax.scatter(xs, ys, zs, marker='o')

    plt.show()

def display_clouds(source, target):
    source = np.array(source).reshape((-1,3))
    target = np.array(target).reshape((-1,3))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(50,250)
    ax.set_ylim(50,250)
    ax.set_zlim(50,250)

    print(source.shape)
    xs, ys, zs = source[:,0], source[:,1], source[:,2]
    ax.scatter(xs, ys, zs, marker='o')

    xt, yt, zt = target[:,0], target[:,1], target[:,2]
    ax.scatter(xt, yt, zt, marker='x')
    ax.legend(["volume","navigation"])

    plt.show()


def main():

    ground_truth = np.fromfile(sys.argv[1], sep=' ').reshape((-1,4,3))
    test = np.fromfile(sys.argv[2], sep=' ').reshape((-1,4,3))
    assert ground_truth.shape == test.shape
    for (source, dest) in zip(ground_truth, test):
        rotation, translation = compute_rigid_transform(source.T, dest.T)
        angle = distance_between_rotation_matrices(np.eye(3), rotation)
        print(angle,end=' ')
        print_colinearity_metrics(dest)

if __name__ == "__main__":
    main()
