import h5py
import numpy as np
import sys
from Acquisition import Acquisition
import os

# This process should be run after technique and offset calibration are both complete
def main():

    # Verify the user gave a path to the file
    if(len(sys.argv)!=3):
        print "Use: PixelGainCalibration.py <path to image.h5> <path to output.h5>"
        return
    try:
        assert os.path.exists(sys.argv[1])
    except:
        print "Input path doesn't exist. Exiting."
        return

    #import the h5 file
    acq = Acquisition(sys.argv[1])

    # Squash the image in the z-direction
    acq.averageAndNormalizeImages(0)

    # Open output file and save out projections
    acq.saveAsFloat(sys.argv[2])


if(__name__ == "__main__"):
    main()
