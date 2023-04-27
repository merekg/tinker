import re
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

# HDF5 acquisition file constants
ITKIMAGE_PATH = 'ITKImage/0/'
VOXELDATA_PATH = ITKIMAGE_PATH + 'VoxelData'

# Dead pixel regex constants
DEAD_PIXEL_SYNTAX_REGEX = '\[[0-9]{0,}:?[0-9]{0,},[0-9]{0,}:?[0-9]{0,}\]\s+'

ACTIVE_PIXEL_PERCENTILE = 95
COLLIMATION_PERCENTILE_THRESHOLD = 0.5

def printHelp():
    print 'usage: python CollimationVerification.py <collimation-acqisition> <dead-pixel-map>'
    print 'Reads in the <collimation-acquisition> file, identifies active pixels (not marked dead in the <dead-pixel-map>) and outputs the pixel value at which the collmator should attenuate intensities'

def validateArguments(arguments):

    try:
        assert len(arguments) == 2
        acquisitionFilePath = arguments[0]
        deadPixelFilePath = arguments[1]
    except:
        printHelp()
        exit()

    return acquisitionFilePath, deadPixelFilePath

def findSlice(line):

    # Check if the line contains a semicolon
    if ':' in line:

        # Split at the semi colon
        (lineLeft, lineRight) = line.split(':')

        # Try get value of left half of line
        try:
            index1 = int(lineLeft)
        except ValueError:
            index1 = None

        # Try get value of right half of line
        try:
            index2 = int(lineRight)
        except ValueError:
            index2 = None

        return slice(index1, index2)

    else:
        return int(line)

def formDeadPixelMap(deadPixelIndicies, mapShape):

    # Create empty map for dead pixels
    deadPixelMap = np.full(mapShape, False)

    # Find dead pixels from file and fill in map
    for lineNumber, line in enumerate(deadPixelIndicies, 1):

        # Identify lines with completely wrong syntax (no brackets or commas)
        if not re.match(DEAD_PIXEL_SYNTAX_REGEX, line):
            print "WARNING: line ", lineNumber, "of", deadPixelFilePath, "contains invalid syntax and will be ignored"
            continue

        # Split by the comman to identify the X and Y part of the dead pixel line
        (lineX, lineY) = line.replace('[', '').replace(']', '').replace('\n', '').split(',')

        # Get the slices for each half
        sliceX = findSlice(lineX)
        sliceY = findSlice(lineY)

        # Set dead pixel map of slice coordinates
        deadPixelMap[sliceX, sliceY] = True

    return np.transpose(deadPixelMap)

def plotDeadPixels(acquisition, deadPixelMap):

    markedPixels = acquisition[0]
    markedPixels[deadPixelMap] = 60000
    figure, axes = plt.subplots(ncols=2, figsize=(30,16))
    axes[0].imshow(deadPixelMap)
    axes[1].imshow(markedPixels)
    plt.show()

def plotIntensityHistogram(acquisition, activePixels):

    figure, axes = plt.subplots(2, figsize=(30,16))
    axes[0].hist(acquisition.flatten(), 100)
    axes[1].hist(activePixels, 100)
    plt.show()

def findCollimationThreshold():

    # Load datasets
    acquisitionFilePath, deadPixelFilePath = validateArguments(sys.argv[1:])
    acquisition = h5py.File(acquisitionFilePath, 'r')[VOXELDATA_PATH][:]
    deadPixelIndicies = open(deadPixelFilePath)

    # Identify active pixels
    deadPixelMap = formDeadPixelMap(deadPixelIndicies, acquisition[0].shape)
    activePixelMap = np.invert(deadPixelMap)
    activePixels = [image[activePixelMap] for image in acquisition]
    activePixels = np.array(activePixels).flatten()

    # Optional plots for debug purposes
    #plotDeadPixels(acquisition, deadPixelMap)
    #plotIntensityHistogram(acquisition, activePixels)

    # Find the collimation threshold for the acquisition
    percentile = np.percentile(activePixels, ACTIVE_PIXEL_PERCENTILE)
    collimationThreshold = COLLIMATION_PERCENTILE_THRESHOLD * percentile

    # Print the computed threshold for colliimation
    print "Collimation Threshold:", collimationThreshold

if __name__ == "__main__":
    findCollimationThreshold()
