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

# The marker is hollow on one side, compensate for this hollowness.
MARKER_CENTER_OFFSET_MM = np.array([-0.6,0,0])

def get_fixture_signature(array):
    signatures = []
    for point in array:
        signatures.append(get_point_signature(point,array))
    return np.array(signatures)

def get_marker_index(marker,cloud):
    signature = get_point_signature(marker,cloud)
    fixture_signature = get_fixture_signature(FIXTURE_ARRAY)
    fixture_signature_min = fixture_signature - SIGNATURE_ERROR
    fixture_signature_max = fixture_signature + SIGNATURE_ERROR
    for i in range(len(fixture_signature_min)):
        if all(signature > fixture_signature_min[i]) and all(signature < fixture_signature_max[i]):
            return i
    print("ERROR: Detected a non-rigid array. Please remove this file and try again.")
    exit()

# Blur the image, then threshold
def process_image(image):
    LOWER_THRESHOLD = 550
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

def get_point_signature(point, cloud):
    distances = []
    for p in cloud:
        if not all(point==p):
            distances.append(np.linalg.norm(point-p))
    return np.array(sorted(distances))

def order_cloud(cloud):
    inds = [1,2,4,0,3]
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
    linearError = np.linalg.norm(error, axis=0)
    print("Volume:")
    print(B.T)
    print("Navigation:")
    print(TA.T)
    print("Difference:")
    print(error.T)

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
        average_marker_centers = order_cloud(np.reshape(average_marker_centers, (-1,3)))
        camera_marker_centers.append(average_marker_centers)
    camera_marker_centers = np.array(camera_marker_centers).reshape((-1, 3))
    print("camera markers:")
    print(np.array([np.linalg.norm(x - np.mean(camera_marker_centers,axis=0)) for x in camera_marker_centers]))
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
        size = header["sizes"]
        extent = spacing * size
        volume_center = [extent[0]/2, extent[1]/2, 0]

        # Clean up the volume
        image_processed = process_image(volume)

        # Connected components 
        image_labeled, nr_objects = ndimage.label(image_processed>= 1)
        nrrd.write("process.nrrd", image_labeled)
        for i in range(2, 1 + nr_objects):
            component = np.where(image_labeled == i)
            marker = np.array(component).T
            r,c = skg.nsphere.nsphere_fit(marker)

            # Rescale and center the volume markers
            current_volume_markers.append(np.array([c[0] * spacing[0], c[1] * spacing[1], c[2] * spacing[2]]) - volume_center)

        # find the direction of the array to apply the hollowness offset
        direction = rotation_between(np.array([1,0,0]),find_array_direction(current_volume_markers))
        marker_offset = direction @ MARKER_CENTER_OFFSET_MM
        current_volume_markers += marker_offset
        
        volume_marker_centers.append((current_volume_markers))

    volume_marker_centers = np.array(volume_marker_centers).reshape((-1,3))
    print("volume markers:")
    print(np.array([np.linalg.norm(x - np.mean(volume_marker_centers,axis=0)) for x in volume_marker_centers]))
    return np.array(volume_marker_centers).reshape((-1,3))

def main():

    testRun, calibration_path, camera_paths, volume_paths = parse_inputs(sys.argv[1:])

    # Navigation points
    camera_marker_centers = get_camera_marker_centers(camera_paths)
    n_camera_markers = len(camera_marker_centers)
    print("Found", n_camera_markers, "markers positions from camera.")

    # volume points
    volume_marker_centers = get_volume_marker_centers(volume_paths)
    n_volume_markers = len(volume_marker_centers)
    print("Found", n_volume_markers, "markers positions from volumes.")

    # Check that the number of markers is the same for each cloud
    assert n_camera_markers is n_volume_markers

    if testRun: # just check a current calibration
        registration_matrix = np.fromfile(calibration_path,sep=' ').reshape((4,4))
        # Perform the check to see how well the points register
        error = check_transformation(camera_marker_centers.T, volume_marker_centers.T, registration_matrix)
        print("Registration error:")
        print(error)
        if max(error) > MAX_ERROR_MM:
            print("WARNING: Max registration error exceeds", MAX_ERROR_MM)
    else: # create a new calibration
        # Compute rigid transform
        rotation, translation = compute_rigid_transform(camera_marker_centers.T, volume_marker_centers.T)
        registration_matrix = to_transform_matrix(rotation, translation.flatten())

        # Perform the check to see how well the points register
        error = check_transformation(camera_marker_centers.T, volume_marker_centers.T, registration_matrix)
        print("Registration error:",np.mean(np.abs(error)))

        # Output files for point and rotatin matrix simulation
        print("Saving calibration as", calibration_path)
        np.savetxt(calibration_path, registration_matrix , delimiter=" ")

if __name__ == "__main__":
    main()
