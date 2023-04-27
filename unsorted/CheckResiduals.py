#! /usr/bin/env python3
import numpy as np
from scipy import ndimage
import nrrd
import sys
import skg
import argparse
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

N_MARKERS = 5
MAX_ERROR_MM = 1.0
MIN_CAMERA_SAMPLES = 100

# The marker is hollow on one side, compensate for this hollowness.
MARKER_CENTER_OFFSET_MM = np.array([0.3,0,0])

# The probe markers based on the tip
PROBE_DEFINITION = np.array([[-159.44,-1.16,-0.71],
                             [-229.29,-1.16,-0.71],
                             [-255.45,-43.83,-0.71],
                             [-263.07,42.27,-0.71],
                             [-304.47,-1.16,-0.71]])

# Blur the image, then threshold
def process_image(image):
    LOWER_THRESHOLD = 1000
    FILTER_STD_DEV = 3
    image_filtered = ndimage.gaussian_filter(image, FILTER_STD_DEV)
    return np.where(image_filtered < LOWER_THRESHOLD, 0, image_filtered)

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

def get_order(cloud):
    centroid = np.mean(cloud, axis=0)
    distances = []
    for marker in cloud:
        distances.append(np.linalg.norm(marker - centroid))

    return distances

def order_cloud(cloud):
    inds = get_order(cloud)
    ordered_cloud = []
    for _,marker in sorted(zip(inds, cloud)):
        ordered_cloud.append(marker)
    return ordered_cloud

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
    print("Projected error (x,y,z):")
    for line in error.T:
        print(*line)
    linearError = np.linalg.norm(error, axis=0)

    b_disp = B
    ta_disp = TA
    #display_clouds(b_disp.T, ta_disp.T)
    # Return the error values
    return linearError

def parse_inputs(args):

    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("-o","--output", help="File path of the output", required=False)
    g.add_argument("-t","--test", help="Run a test of the current calibration", required=False)
    parser.add_argument("-c","--cameraPoints", nargs='+',help="Files containing camera points", required=True)
    parser.add_argument('-v','--volumes', nargs='+', help="File paths of volumes [list]", required=True)

    parsed_list = parser.parse_args(args).__dict__

    testRun = parsed_list["test"] is not None

    # change relative paths to absolute paths
    calibration_path = os.path.abspath(parsed_list["output"]) if not testRun else os.path.abspath(parsed_list["test"])
    camera_paths = (parsed_list["cameraPoints"])
    volume_paths = parsed_list["volumes"]
    abs_volume_paths = []
    abs_camera_paths = []
    for path in volume_paths:
        abs_volume_paths.append(os.path.abspath(path))
    for path in camera_paths:
        abs_camera_paths.append(os.path.abspath(path))

    return testRun, calibration_path, abs_camera_paths, abs_volume_paths

def get_camera_marker_centers(camera_paths):
    camera_marker_centers = []

    for camera_path in camera_paths:
        print(camera_path)
        nav_file = open(camera_path,'r')
        tool_poses = []
        for line in nav_file:
            if "nan" in line:
                continue
            tool_poses.append(np.array(line.replace(' ','').strip()[:-1].split(",")).astype(np.float))

        # check that there are enough samples. if no, don't continue
        if(len(tool_poses) < MIN_CAMERA_SAMPLES):
            print("Error: too few camera samples for file", camera_path)
            print("Please remove file and collect another sample")
            exit()

        # Average the samples
        average_tool_pose = np.mean(np.array(tool_poses), axis=0)

        # The tool data comes in as [x,y,z,qx,qy,qz,qw]. Use this to determine the marker poses
        probe_rotation = R.from_quat(np.array([average_tool_pose[3],average_tool_pose[4],average_tool_pose[5],average_tool_pose[6]])).as_matrix()
        probe_translation = average_tool_pose[:3]
        marker_poses = np.array([probe_rotation @ marker + probe_translation for marker in PROBE_DEFINITION])

        # Put the markers in a well-defined order to be able to match up with the volume markers
        ordered_marker_poses = order_cloud(np.reshape(marker_poses, (-1,3)))
        camera_marker_centers.append(ordered_marker_poses)
    camera_marker_centers = np.array(camera_marker_centers).reshape((-1, 3))
    return camera_marker_centers

def find_array_direction(points):
    points = np.array(points).reshape((-1,3)).T
    
    # subtract out the centroid and take the SVD
    svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))

    # Extract the left singular vectors
    left = svd[0]

    # the corresponding left singular vector is the normal vector of the best-fitting plane
    return left[:, -1]

def rotation_between(v1,v2):
    a, b = (v1 / np.linalg.norm(v1)).reshape(3), (v2 / np.linalg.norm(v2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def get_volume_marker_centers(volume_paths):
    volume_marker_centers = []
    for volume_path in volume_paths:
        print(volume_path)
        current_volume_markers = []
        # Read in the volume from the console
        volume, header = nrrd.read(volume_path)
        spacing = header["spacings"]
        size = header["sizes"] - 1
        extent = spacing * size
        volume_center = [extent[0]/2, extent[1]/2, 0]

        # Clean up the volume
        image_processed = process_image(volume)

        # Connected components 
        image_labeled, nr_objects = ndimage.label(image_processed>= 1)
        # There may be the metal of the probe shaft in the image
        if not(nr_objects == N_MARKERS+1):
            print("Error: Expected", N_MARKERS, "markers and one probe shaft, found", nr_objects, "objects.")
            print("Make sure the probe is well-placed in the volume, and take this data pair again.")
            exit()
        for i in range(1, 1 + nr_objects):
            component = np.where(image_labeled == i)
            marker = np.array(component).T
            r,c = skg.nsphere.nsphere_fit(marker)

            # Check that the object found appears to be a sphere
            if r < 8:
                # Rescale and center the volume markers
                current_volume_markers.append(np.array([c[0] * spacing[0], c[1] * spacing[1], c[2] * spacing[2]]) - volume_center)

        # find the direction of the array to apply the hollowness offset
        direction = rotation_between(np.array([1,0,0]),find_array_direction(current_volume_markers))
        marker_offset = direction @ MARKER_CENTER_OFFSET_MM
        current_volume_markers += marker_offset
        
        volume_marker_centers.append(order_cloud(current_volume_markers))

    return np.array(volume_marker_centers).reshape((-1,3))

def main():

    # Navigation points
    camera_marker_centers = np.fromfile(sys.argv[1],sep=' ').reshape((-1,3))
    n_camera_markers = len(camera_marker_centers)
    print("Found", n_camera_markers, "markers positions from camera.")
    print(camera_marker_centers)

    # volume points
    volume_marker_centers = np.fromfile(sys.argv[2],sep=' ').reshape((-1,3))
    n_volume_markers = len(volume_marker_centers)
    print("Found", n_volume_markers, "markers positions from volumes.")
    print(volume_marker_centers)

    # Check that the number of markers is the same for each cloud
    assert n_camera_markers is n_volume_markers

    # Compute rigid transform
    rotation, translation = compute_rigid_transform(camera_marker_centers.T, volume_marker_centers.T)
    registration_matrix = to_transform_matrix(rotation, translation.flatten())

    # Perform the check to see how well the points register
    error = check_transformation(camera_marker_centers.T, volume_marker_centers.T, registration_matrix)
    print("Max registration error:",max(error))

if __name__ == "__main__":
    main()
