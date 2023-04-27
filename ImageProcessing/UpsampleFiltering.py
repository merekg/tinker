import sys
import h5py
import nrrd
import os
import numpy as np

TEEM_UNU_COMMAND = 'teem-unu resample -s 512 512 512 -k black:3'
NRRD_TMP_PATH = '/tmp/recon.tmp.nrrd'
NRRD_VIEW_PATH = '/tmp/recon.view.nrrd'

if __name__ == '__main__':

    # Read input arguments to get input file name (fail if no arguments)
    try:
        filePath = sys.argv[1]
    except:
        print 'No input path'
        sys.exit(1)

    # Extract path, file base, and extension
    fileFolder, fileName = os.path.split(filePath)
    fileBase, fileExt = os.path.splitext(fileName)

    # Convert input file to nrrd
    with h5py.File(filePath, 'r') as inputFile:
        nrrd.write(NRRD_TMP_PATH, inputFile['/ITKImage/0/VoxelData'][:], {})

    # Upsample nrrd file using teem_unu
    resampleCommand = TEEM_UNU_COMMAND + ' -i ' + NRRD_TMP_PATH + ' -o ' + NRRD_VIEW_PATH
    os.system(resampleCommand)

    # Convert output file back to H5
    viewPath = os.path.join(fileFolder, fileBase + '.view' + fileExt)
    with h5py.File(viewPath) as outputFile:

        # Write MITK parameters
        outputFile.create_dataset('/ITKImage/0/Dimension', data=np.array([512,512,512]))
        outputFile.create_dataset('/ITKImage/0/Directions', data=np.identity(3))
        outputFile.create_dataset('/ITKImage/0/Origin', data=np.zeros(3))
        outputFile.create_dataset('/ITKImage/0/Spacing', data=np.array([0.5859375,0.29296875,0.234375]))

        # Write image data and scale
        volume, imageDataHeader = nrrd.read(NRRD_VIEW_PATH)
        outputFile.create_dataset('/ITKImage/0/VoxelData', data=volume)
        outputFile['/ITKImage/0/VoxelData'].attrs.create('scale', 10922.500000)
        outputFile.create_dataset('/ITKImage/0/MetaData/P', data=np.ones(2))
