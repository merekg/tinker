import numpy as np
from scipy import ndimage
import nrrd
import sys
import skg
import argparse
import os

N_MARKERS = 6
VIEWER_VOLUME_CENTER = [150, 150, 0]

## TODO: REMOVE THE FOLLOWING PACKAGES
import matplotlib.pyplot as plt

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

def order_cloud(cloud):
    mean = np.array(cloud).mean(axis=0)
    distances = {}

    for i in range(len(cloud)):
        distances[np.linalg.norm(np.absolute(cloud[i] - mean))] = i
    sorted_indices = []
    for tup in sorted(distances.items()):
        sorted_indices.append(tup[1])
    return cloud[sorted_indices]

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

def parse_inputs(args):

    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--output", help="File path of the output", required=True)
    parser.add_argument("-c","--cameraPoints", nargs='+',help="Files containing camera points", required=True)
    parser.add_argument('-v','--volumes', nargs='+', help="File paths of volumes [list]", required=True)

    parsed_list = parser.parse_args(args).__dict__

    # change relative paths to absolute paths
    out_path = os.path.abspath(parsed_list["output"])
    camera_paths = (parsed_list["cameraPoints"])
    volume_paths = parsed_list["volumes"]
    abs_volume_paths = []
    abs_camera_paths = []
    for path in volume_paths:
        abs_volume_paths.append(os.path.abspath(path))
    for path in camera_paths:
        abs_camera_paths.append(os.path.abspath(path))


    return out_path, abs_camera_paths, abs_volume_paths

def main():

    out_path, camera_paths, volume_paths = parse_inputs(sys.argv[1:])

    # Navigation points
    nav_marker_centers = []
    for camera_path in camera_paths:
        f = open(camera_path, 'r')
        average_marker_centers = np.zeros_like(np.array(f.readlines()[0].strip()[:-1].split(",")).astype(np.float))
        nav_file = open(camera_path,'r')
        n_samples = 0
        for line in nav_file:
            n_samples += 1
            average_marker_centers = average_marker_centers + np.array(line.replace(' ','').strip()[:-1].split(",")).astype(np.float)

        average_marker_centers /= n_samples
        average_marker_centers = np.reshape(average_marker_centers, (-1,3))
        nav_marker_centers.append(average_marker_centers)
    nav_marker_centers = np.array(nav_marker_centers).reshape((-1, 3))
    n_camera_markers = len(nav_marker_centers)
    print("Found", n_camera_markers, "markers positions from camera.")

    # volume points
    volume_raw_markers = []
    spacings = []
    for volume_path in volume_paths:
        # Read in the volume from the console
        volume, header = nrrd.read(volume_path)
        spacings.append(header["spacings"][0])

        # Clean up the volume
        image_processed = process_image(volume)

        # Connected components 
        image_labeled, nr_objects = ndimage.label(image_processed>= 1)
        print(volume_path + ": Number of markers found: " + str(nr_objects))
        if nr_objects is not N_MARKERS:
            print("WARNING: Expected", N_MARKERS, "markers, found", nr_objects, "objects.")
        for i in range(1, 1 + nr_objects):
            r = np.where(image_labeled == i)
            volume_raw_markers.append(r)

    # Check that all spacings are the same
    spacings = np.unique(spacings)
    assert len(spacings) is 1
    spacing = spacings[0]

    # Find the center of each connected component
    volume_marker_centers = []
    for m in volume_raw_markers:
        marker = np.array(m).T
        r,c = skg.nsphere.nsphere_fit(marker)
        volume_marker_centers.append(c * spacing - VIEWER_VOLUME_CENTER)
    volume_marker_centers = np.array(volume_marker_centers)
    n_volume_markers = len(volume_marker_centers)
    print("Found", n_volume_markers, "markers positions from volumes.")
    assert n_camera_markers is n_volume_markers
    print(volume_marker_centers)

    # order the cloud to register them together
    ordered_camera_points = order_cloud(nav_marker_centers)
    ordered_volume_points = order_cloud(volume_marker_centers)

    # Compute rigid transform
    rotation, translation = compute_rigid_transform(ordered_camera_points.T, ordered_volume_points.T)
    registration_matrix = to_transform_matrix(rotation, translation.flatten())

    # Perform the check to see how well the points register
    error = check_transformation(ordered_camera_points.T, ordered_volume_points.T, registration_matrix)
    print("Registration error:",error)

    # Output files for point and rotatin matrix simulation
    print("Saving calibration as", out_path)
    np.savetxt(out_path, registration_matrix , delimiter=" ")

if __name__ == "__main__":
    main()
