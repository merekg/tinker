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
MARKER_CENTER_OFFSET_MM = np.array([0.5,0,0])

# Blur the image, then threshold
def process_image(image):
    LOWER_THRESHOLD = 550
    FILTER_STD_DEV = 3
    image_filtered = ndimage.gaussian_filter(image, FILTER_STD_DEV)
    return np.where(image_filtered < LOWER_THRESHOLD, 0, image_filtered)

def main():
    
    volume_path = sys.argv[1]

    current_volume_markers = []
    # Read in the volume from the console
    volume, header = nrrd.read(volume_path)
    spacing = header["spacings"]
    size = header["sizes"]
    extent = spacing * size
    volume_center = [extent[0]/2, extent[1]/2, 0]

    # Clean up the volume
    image_processed = process_image(volume)
    nrrd.write("/home/nview/Desktop/workspace/2022.08.16_hollowMarkerStudy/original.nrrd",volume, header)
    nrrd.write("/home/nview/Desktop/workspace/2022.08.16_hollowMarkerStudy/testImage.nrrd",image_processed, header)

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
        current_volume_markers.append(np.array([c[0] * spacing[0], c[1] * spacing[1], c[2] * spacing[2]]) - volume_center)

    print(np.array(current_volume_markers))

if __name__ == "__main__":
    main()
