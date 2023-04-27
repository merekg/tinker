import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from collections import defaultdict

# Define constants for application
ROI_RATIO = 1.0 / 3.0 # Ratio of full image size for ROI (center 1/3rd of image)

# Return the output file path and an array of all input paths
def parseInputs(args):
    # check that there are a reasonable number of inputs
    if(len(args) < 2):
        print("Use: python TechniqueCalibration.py <input .h5 files> <output file path>")
        exit()

    return [args[-1], args[:-1]]

# Correct the acquisitions for dark image
def correctAcquisitions(inputAcquisitionPaths):
    # Array of the filepaths to the corrected images
    correctedPaths = ["/tmp/" + os.path.split(name)[1] for name in inputAcquisitionPaths]

    # Apply needed corrections to each file
    for uncorrected, corrected in zip(inputAcquisitionPaths, correctedPaths):
        subprocess.call(["ImageCorrector",
                        "-i", uncorrected,
                        "-u", corrected,
                        "-o"])

    return correctedPaths

# Return a dictionary of all the techniques and all the acquisition files that use each technique
def sortAcquisitions(acquisitionPaths):
    # Dictionary to hold our results
    techniqueSlope = defaultdict(list)
    # determine the techniques of all the acquisitions
    for path in acquisitionPaths:
        acq = h5py.File(path, 'r')
        kv = acq["/ITKImage/0/MetaData/metaDataTable"][0][2]
        ma = acq["/ITKImage/0/MetaData/metaDataTable"][0][4]
        technique = str(kv) + " " + str(ma)
        techniqueSlope[technique].append(path)

    return techniqueSlope

# Return the slope of the given acquisitions at the given technique
def techniqueRegression(correctedPaths, technique, outdir):
    imageStacks = []
    imageMetaDataTables = []

    for imageStackPath in correctedPaths:

        imageStackFile = h5py.File(imageStackPath, 'r')
        imageStacks.append(imageStackFile['/ITKImage/0/VoxelData'][:])
        imageMetaDataTables.append(imageStackFile['/ITKImage/0/MetaData/metaDataTable'][:])

    kVActual = []
    mAActual = []
    kVSet = []
    mASet = []
    intensity = []

    # Compute ROI size based on ROI ration of image size
    imageSize = imageStacks[0].shape
    ROISliceX = slice(int(imageSize[1] * ROI_RATIO), int(imageSize[1] * 2 * ROI_RATIO))
    ROISliceY = slice(int(imageSize[2] * ROI_RATIO), int(imageSize[2] * 2 * ROI_RATIO))

    #TODO: Make the loading of images an extend operation rather than append to remove need to rearrange data

    for stack, table in zip(imageStacks, imageMetaDataTables):

        kVActual.extend(np.array([row[1] for row in table]))
        mAActual.extend(np.array([row[3] for row in table]))
        kVSet.extend(np.array([row[2] for row in table]))
        mASet.extend(np.array([row[4] for row in table]))
        intensity.extend(np.mean(stack[:,ROISliceX,ROISliceY], axis=(1,2)))

    # Correct intensity for mAActual (NOTE: acquisitions could be taken with mAActual corrected instead)
    mACorrectedIntensity = np.array(intensity) * np.array(mASet) / np.array(mAActual)

    # Perform linear regression
    x = kVActual
    y = mACorrectedIntensity
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]

    # Compute normalized slope based on kV set value
    intensitySet = m * kVSet[0] + b
    slope = m / intensitySet

    # Compute standard deviation of error of linear regression
    kVCorrectedIntensity = mACorrectedIntensity - (np.array(kVActual) * m + b)
    intensityStdDev = np.std(kVCorrectedIntensity)

    # Print final results
    print "kV-set: " + str(kVSet[0])
    print "Slope: " + str(slope)
    # Create subplots
    figure, axes = plt.subplots(2, figsize=(20,16))

    # Plot intensity against kV
    axes[0].plot(kVActual, mACorrectedIntensity, 'ro')
    axes[0].plot(np.linspace(min(kVActual), max(kVActual)), np.linspace(min(kVActual), max(kVActual)) * m + b)
    axes[0].set(ylabel='Average ROI Intensity')
    axes[0].set(xlabel='kV')
    #axes[0].set_ylim((4000, 5500))

    # Plot error against kV
    axes[1].plot(kVActual, kVCorrectedIntensity, 'ro')
    axes[1].plot([min(kVActual), max(kVActual)], [0.0, 0.0])
    axes[1].set(ylabel='Linear Regression Error')
    axes[1].set(xlabel='kV')
    axes[1].set_ylim((-50, 50))

    # Save plot and display
    plt.savefig(os.path.join(outdir, str(technique) + 'kV_plot.png'))
    #plt.show()
    return str(slope)

# Save the values in techniqueSlope to outputPath
def writeCalibrationFile(outputPath, techniqueSlope):
    print("Saving to " +str(outputPath) + "...")
    out = open(outputPath, "w+")
    for kv, slope in techniqueSlope.items():
        line = str(kv) + " " + str(slope) + "\n"
        out.write(line)
    out.close()


def main():

    outputFilePath, inputPaths = parseInputs(sys.argv[1:])
    correctedAcqPaths = correctAcquisitions(inputPaths)

    techniqueMap = sortAcquisitions(correctedAcqPaths)

    # a dictionary to hold all the outputs
    techniqueSlopes = {}
    for technique, acqPaths in techniqueMap.items():
        techniqueSlopes[str(technique)] = techniqueRegression(acqPaths, technique,os.path.split(outputFilePath)[0])

    writeCalibrationFile(outputFilePath, techniqueSlopes)


if __name__ == "__main__":
    main()
