import h5py
import sys
import re
import os

# Scale parameter constants
SCALE_VALUE = '10922.5'
SCALE_PATH = '/ITKImage/0/VoxelData'
SCALE_ATTR = 'scale'

# Insert PnO for a given file path
def insertScale(inputPath):

    # Display path image scale is being applied to
    print "Writing scale for: " + inputPath

    # Get image name from image path
    inputImage = os.path.basename(inputPath)

    # Attempt to open image
    with h5py.File(inputPath, 'r+') as inputFile:

        # Attempt to delete scale attribute
        try:
            del inputFile[SCALE_PATH].attrs[SCALE_ATTR]
        except:
            pass

        # Write correct scale attribute
        inputFile[SCALE_PATH].attrs.create(SCALE_ATTR, SCALE_VALUE)

if __name__ == '__main__':
    for inputPath in sys.argv[1:]:
        insertScale(inputPath)
