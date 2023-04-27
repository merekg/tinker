import h5py
import sys
import re
import os

# Define marker constants
INITIAL_MARKER = 'i-'
PLUS_MARKER = 'i+'

# Define PnO constants
INITIAL_PNO = [-3.41, 63.07, 11.82, 30.71, 1.40, -0.21]
PLUS_PNO = [-6.70, -56.52, 20.73, -28.95, 1.61, -0.71]
SINGLE_PNO = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Insert PnO for a given file path
def insertPnO(inputPath):

    # Print image PnO is being added to
    print "Writing PnO for: " + inputPath

    # Get image name from image path
    inputImage = os.path.basename(inputPath)

    # Indentify image type
    if INITIAL_MARKER in inputImage:
        imagePnO = INITIAL_PNO
    elif PLUS_MARKER in inputImage:
        imagePnO = PLUS_PNO
    else:
        imagePnO = SINGLE_PNO

    # Attempt to open image
    with h5py.File(inputPath, 'r+') as inputFile:

        # Attempt to write PnO, pass if alreedy written
        try:
            inputFile.create_dataset('/ITKImage/0/MetaData/imagePnO', data=imagePnO)
        except:
            print "    imagePnO already exists"
            pass

# Insert PnO for all given paths (supporting wild card)
def insertPnOAll():
    for inputPath in sys.argv[1:]:
        insertPnO(inputPath)

if __name__ == '__main__':
    insertPnOAll()
