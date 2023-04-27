import sys
import os
import shutil
import h5py
import scipy.ndimage as ndi

FILTER_KERNEL_SIZE = 1

def main(): 

    # Read input arguments to get path of file name
    if(len(sys.argv) !=3):
        print 'No input/output file path given.'
        print 'Use: FilterSignalInAirProjections <inputPath> <outputPath>'
        sys.exit(1)

    inputPath = sys.argv[1]
    outputPath = sys.argv[2]
    # Extract folder, file base, and extension from input path
    inputFolder, inputName = os.path.split(inputPath)
    inputBase, inputExt = os.path.splitext(inputName)

    with h5py.File(inputPath, 'r') as inputFile:

        # Get projection data from input file
        projections = inputFile['ITKImage/0/VoxelData'][:]

    # Filter each projection
    filteredProjections = ndi.median_filter(projections, size=(3,3,3))

    # Formulate output path and copy input file to output path
    shutil.copyfile(inputPath, outputPath)

    # Open output file and save out projections
    with h5py.File(outputPath, 'r+') as outputFile:

        # Write filtered projections to output file voxel data
        voxelData = outputFile['ITKImage/0/VoxelData']
        voxelData[...] = filteredProjections

if __name__ == '__main__':
    main()
