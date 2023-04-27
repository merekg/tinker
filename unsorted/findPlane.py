import numpy as np
import sys
import nrrd
from scipy import ndimage
import skg

N_MARKERS = 6
VIEWER_VOLUME_CENTER = [150, 150, 0]

def process_image(image):
    LOWER_THRESHOLD = 550
    FILTER_STD_DEV = 3
    image_filtered = ndimage.gaussian_filter(image, FILTER_STD_DEV)
    return np.where(image_filtered < LOWER_THRESHOLD, 0, image_filtered)

def get_marker_centers(volume_paths):
    volume_marker_centers = []
    for volume_path in volume_paths:
        current_volume_markers = []
        # Read in the volume from the console
        volume, header = nrrd.read(volume_path)
        spacing = header["spacings"][0]

        # Clean up the volume
        image_processed = process_image(volume)

        # Connected components 
        image_labeled, nr_objects = ndimage.label(image_processed>= 1)
        if nr_objects is not N_MARKERS:
            print("Error: Expected", N_MARKERS, "markers, found", nr_objects, "objects.")
            exit()
        for i in range(1, 1 + nr_objects):
            component = np.where(image_labeled == i)
            marker = np.array(component).T
            r,c = skg.nsphere.nsphere_fit(marker)

            # Rescale and center the volume markers
            current_volume_markers.append(c * spacing - VIEWER_VOLUME_CENTER)

        # report the current normal vector
        points = np.array(current_volume_markers).T
        svd = np.linalg.svd(points- np.mean(points, axis=1, keepdims=True))

        left = svd[0]
        print(left[:,-1])


    return np.array(volume_marker_centers).reshape((-1,3))

average_marker_centers = get_marker_centers(sys.argv[1:])

