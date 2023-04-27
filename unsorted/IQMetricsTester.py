from matplotlib import pyplot as plt
import scipy.stats as st
import numpy as np
import subprocess
import shutil
import h5py
import os
import re

# Define image parameters
PLANE_RESOLUTION_MM = 1.0 # mm per voxel
SLICE_RESOLUTION_MM = 1.0 # mm per voxel
VOXEL_SIZE_MM = (PLANE_RESOLUTION_MM, PLANE_RESOLUTION_MM, SLICE_RESOLUTION_MM)
SAVE_SCALE_FACTOR = 8192

# Define CTP404 module parameters
CTP404_HEIGHT_MM = 25.0 # mm
CTP404_HEIGHT_VX = int(CTP404_HEIGHT_MM / SLICE_RESOLUTION_MM) # voxels
CTP404_RADIUS_MM = 75.0 # mm
CTP404_RADIUS_VX = int(CTP404_RADIUS_MM / SLICE_RESOLUTION_MM) # voxels
CTP404_CENTER_MM = [CTP404_RADIUS_MM, CTP404_RADIUS_MM, CTP404_HEIGHT_MM/2]
CTP404_CENTER_VX = (np.array(CTP404_CENTER_MM) / np.array(VOXEL_SIZE_MM)).astype(int)
CTP404_TEFLON_CENTER_MM = np.array(CTP404_CENTER_MM) + np.array([25, 50, 0])
CTP404_TEFLON_CENTER_VX = (CTP404_TEFLON_CENTER_MM * np.array(VOXEL_SIZE_MM)).astype(int)
CTP404_TEFLON_RADIUS_MM = 6.0 # mm
CTP404_TEFLON_RADIUS_VX = int(CTP404_TEFLON_RADIUS_MM / PLANE_RESOLUTION_MM) # voxels
CTP404_TEFLON_DENSITY = 1100 #TODO: use HU or CM^-1 instead of meaningless values
CTP404_MODULE_DENSITY = 100 #TODO: use HU or CM^-1 instead of meaningless values

# Define CTP486 module parameters
CTP486_HEIGHT_MM = 70.0 # mm
CTP486_HEIGHT_VX = int(CTP486_HEIGHT_MM / SLICE_RESOLUTION_MM) # voxels
CTP486_RADIUS_MM = 75.0 # mm
CTP486_RADIUS_VX = int(CTP486_RADIUS_MM / SLICE_RESOLUTION_MM) # voxels
CTP486_MODULE_DENSITY = 100 #TODO: use HU or CM^-1 instead of generic value
CTP486_BACKGROUND_WIDTH_VX = CTP486_RADIUS_VX * 2
CTP486_BACKGROUND_HEIGHT_VX = CTP486_HEIGHT_VX
CTP486_CENTER_VX = [CTP486_BACKGROUND_WIDTH_VX/2,
                    CTP486_BACKGROUND_WIDTH_VX/2,
                    CTP486_BACKGROUND_HEIGHT_VX/2]
CTP486_CENTER_MM = np.array(CTP486_CENTER_VX) * np.array(VOXEL_SIZE_MM)

# Define ZST module parameters
ZST_BACKGROUND_DENSITY = 100.0 #TODO: Choose meaningful densities (HU or cm-1)
ZST_POINT_DENSITY = 1100.0 #TODO: Choose meaningful densities (HU or cm-1)
ZST_SLICE_COUNT = 11
ZST_HEIGHT_MM = 250.0 # mm
ZST_WIDTH_MM = 2500.0 # mm
ZST_HEIGHT_VX = int(ZST_HEIGHT_MM / SLICE_RESOLUTION_MM) # voxels
ZST_WIDTH_VX = int(ZST_WIDTH_MM / PLANE_RESOLUTION_MM) # voxels
ZST_POINT_WIDTH = 2.5 # mm
ZST_NOISE_STD = 5
ZST_BB_DIAMETER_MM = 2 # mm

# Define CNR parameters (in addition to CTP404 parameters)
CNR_NOISE_STD = 100 / np.sqrt(2) # Chosen to give a CNR of 10
CNR_OUTPUT_RE = 'CNR is\:' # Regular expression for CNR output
CNR_ZERO_CHECK = (0.0, 0.1) # Range of acceptable values for CNR zero test
CNR_INFINITE_CHECK = (float('inf'), float('inf')) # Inifinite range for check
CNR_10_CHECK = (9, 11) # Range of acceptable values for CNR 10 test

# Define UI parameters (in addition to CTP486 module parameters)
UI_EDGE_RADIUS_MM = 15.0 # mm
UI_EDGE_RADIUS_VX = int(UI_EDGE_RADIUS_MM / PLANE_RESOLUTION_MM)
UI_EDGE_DENSITY = 1.4 * CTP486_MODULE_DENSITY
UI_POINT_RADIUS_MM = 60.0
UI_POINT_RADIUS_VX = int(UI_POINT_RADIUS_MM / PLANE_RESOLUTION_MM)
UI_COORDINATES = [CTP486_RADIUS_VX, CTP486_RADIUS_VX, CTP404_HEIGHT_VX/2]
UI_NOISE_STD = CTP486_MODULE_DENSITY / 10.0
UI_OUTPUT_RE = 'Uniformity Index Max \[\%\]\:' # Regular expression for UI output
UI_IDEAL_CHECK = (0, 0)
UI_40_CHECK = (40, 40)
UI_NOISE_CHECK = (33, 47)

# Define MTF parameters (in addition to CTP404 parameters)
MTF_NOISE_STD = 100 / np.sqrt(2)
MTF_OUTPUT_RE = 'Spatial frequency at 0\.5MTF \[mm\^\-1\]\:' # Regular expression for MTF output
MTF_IDEAL_CHECK = (0.48, 0.5)
MTF_MEASURED_CHECK = (0.18, 0.22)
MTF_NOISE_CHECK = (0.18, 0.22)

# Define ZST parameters
ZST_OUTPUT_RE = 'The FWHM \[mm\]\:' # Regular expression for ZST output
ZST_MEASURED_CHECK = (0.48, 0.52) # Adjusted to account for a 2mm BB
ZST_NOISE_CHECK = (0.4, 0.6) # Adjusted to account for a 2mm BB

# Define NPS parameters
NPS_NOISE_STD = 100 # TODO: Choose meaningful units (HU or cm-1)
NPS_SAMPLE_COUNT = 10 # Number of sample volumes to run NPS on
NPS_SINE_FREQ = 0.25 # Cycles / mm for sine wave test

# Define names and flags for applications
INPUT_FILE_FOLDER = '/tmp/'
INPUT_FILE_BASE = INPUT_FILE_FOLDER + 'input'
INPUT_FILE_EXTENSION = '.h5'
INPUT_FILE_NAME = INPUT_FILE_BASE + INPUT_FILE_EXTENSION
PYTHON = 'python'
METRICS_APPLICATION = 'IQ_Metrics.py'
METRICS_CNR_FLAG = '--cnr'
METRICS_MTF_FLAG = '--mtf2d'
METRICS_UI_FLAG = '--ui'
METRICS_ZST_FLAG = '--zst'
METRICS_NPS_FLAG = '--nps'
METRICS_NPS3D_FLAG = '--nps3d'

def saveVolumeHdf5(name, volume, dimensions, spacing, scale):

    # Open HDF5 file and write data
    with h5py.File(name, 'w') as output:

        # Write MITK parameters
        output.create_dataset('/ITKImage/0/Dimension', data=dimensions)
        output.create_dataset('/ITKImage/0/Directions', data=np.identity(3))
        output.create_dataset('/ITKImage/0/Origin', data=np.zeros(3))
        output.create_dataset('/ITKImage/0/Spacing', data=spacing)

        # Write image data and scale
        volume = np.transpose(volume, (2, 1, 0))
        output.create_dataset('/ITKImage/0/VoxelData', data=volume)
        output['/ITKImage/0/VoxelData'].attrs.create('scale', 10922.500000)
            # TODO: figure out scale output
        output.create_dataset('/ITKImage/0/MetaData/P', data=np.ones(2))
            # TODO: add useful metadata

def checkResult(output, regex, check, text=None):

    # Find line containting `text` then find value in line
    string = regex.replace("\\", "")
    line = re.search(regex + ".*", output).group()[len(string):]
    value = re.search('(\d+\.?\d*)|(\-?inf)|(nan)', line)
    result = float(value.group())

    # Check if result is within [minimum, maximum]
    report = (check[0] <= result) and (result <= check[1])

    # Rreport pass/fail with errors
    if report:
        print "[PASS] " + text
    else:
        print "[FAIL] " + text
        print "\tResult", result, "should be within", list(check)

    return report

def runCNRTest():

    # Formulate parameters for CNR
    dimensions = np.array([CTP404_RADIUS_VX * 2,
                           CTP404_RADIUS_VX * 2,
                           CTP404_HEIGHT_VX])
    moduleX = np.linspace(-CTP404_RADIUS_MM, CTP404_RADIUS_MM, dimensions[0])
    moduleY = np.linspace(-CTP404_RADIUS_MM, CTP404_RADIUS_MM, dimensions[1])
    moduleMask = moduleX[:,None]**2 + moduleY[None,:]**2 < CTP404_RADIUS_MM**2
    moduleMask = np.repeat(moduleMask[:,:,None], dimensions[2], axis=2)
    plugX = np.linspace(-CTP404_TEFLON_CENTER_MM[0],
                        2*CTP404_CENTER_MM[0] - CTP404_TEFLON_CENTER_MM[0],
                        dimensions[0])
    plugY = np.linspace(-CTP404_TEFLON_CENTER_MM[1],
                        2*CTP404_CENTER_MM[1] - CTP404_TEFLON_CENTER_MM[1],
                        dimensions[1])
    plugMask = plugX[:,None]**2 + plugY[None,:]**2 < CTP404_TEFLON_RADIUS_MM**2
    plugMask = np.repeat(plugMask[:,:,None], dimensions[2], axis=2)
    volume = np.zeros(dimensions)
    volume[moduleMask] = CTP404_MODULE_DENSITY
    volume[plugMask] = CTP404_TEFLON_DENSITY
    noise = np.random.normal(0, CNR_NOISE_STD, dimensions.astype(int))

    # Formulate command line call for CNR
    plugCoordinates = [str(int(pnt)) for pnt in CTP404_TEFLON_CENTER_VX]
    moduleCoordinates = [str(int(pnt)) for pnt in CTP404_CENTER_VX]
    coordinates = plugCoordinates + moduleCoordinates
    command = [PYTHON, METRICS_APPLICATION, INPUT_FILE_NAME, METRICS_CNR_FLAG]

    # Run IQ_Metrics script for zero contrast scenario
    saveVolumeHdf5(INPUT_FILE_NAME, noise, dimensions, VOXEL_SIZE_MM, SAVE_SCALE_FACTOR)
    output = subprocess.check_output(command + coordinates)
    checkResult(output, CNR_OUTPUT_RE, CNR_ZERO_CHECK, "CNR zero contrast")

    # Run IQ_Metrics script for infinite contrast scenario
    saveVolumeHdf5(INPUT_FILE_NAME, volume, dimensions, VOXEL_SIZE_MM, SAVE_SCALE_FACTOR)
    output = subprocess.check_output(command + coordinates, stderr=open(os.devnull, 'w'))
    checkResult(output, CNR_OUTPUT_RE, CNR_INFINITE_CHECK, "CNR infinite contrast")

    # Run IQ_Metrics script for measured contrast scenario
    volume = volume + noise
    saveVolumeHdf5(INPUT_FILE_NAME, volume, dimensions, VOXEL_SIZE_MM, SAVE_SCALE_FACTOR)
    output = subprocess.check_output(command + coordinates)
    checkResult(output, CNR_OUTPUT_RE, CNR_10_CHECK, "CNR 10 contrast")

    # Clean up test file
    os.remove(INPUT_FILE_NAME)

def runUITest():

    # Formulate command line call for UI
    coordinates = [str(int(point)) for point in UI_COORDINATES]
    command = [PYTHON, METRICS_APPLICATION, INPUT_FILE_NAME, METRICS_UI_FLAG]
    dimensions = (CTP486_RADIUS_VX*2, CTP486_RADIUS_VX*2, CTP486_HEIGHT_VX)
    volume = np.zeros(dimensions)
    noise = np.random.normal(0, CNR_NOISE_STD, dimensions)

    # Generate perfect uniformity dataset
    # Run IQ_Metrics script for perfect uniformity dataset
    for (x, y, z), value in np.ndenumerate(volume):
        if (x - CTP486_RADIUS_VX)**2 + (y - CTP486_RADIUS_VX)**2 <= CTP486_RADIUS_VX**2:
            volume[x][y][z] = UI_EDGE_DENSITY
    saveVolumeHdf5(INPUT_FILE_NAME, volume, dimensions, VOXEL_SIZE_MM, SAVE_SCALE_FACTOR)
    output = subprocess.check_output(command + coordinates)
    checkResult(output, UI_OUTPUT_RE, UI_IDEAL_CHECK, "UI perfect uniformity")

    # Generate perfect 40 UI dataset (starting with previous dataset)
    # Run IQ_Metrics script for perfect uniformity dataset
    for (x, y, z), value in np.ndenumerate(volume):
        if (x - CTP486_RADIUS_VX)**2 + (y - CTP486_RADIUS_VX)**2 <= (UI_EDGE_RADIUS_VX)**2:
            volume[x][y][z] = CTP486_MODULE_DENSITY
    saveVolumeHdf5(INPUT_FILE_NAME, volume, dimensions, VOXEL_SIZE_MM, SAVE_SCALE_FACTOR)
    output = subprocess.check_output(command + coordinates)
    checkResult(output, UI_OUTPUT_RE, UI_40_CHECK, "UI 40 without noise")

    # Generate noisy 40 UI dataset (starting with previous dataset)
    # Run IQ_Metrics script for perfect uniformity dataset
    volume = volume + noise
    saveVolumeHdf5(INPUT_FILE_NAME, volume, dimensions, VOXEL_SIZE_MM, SAVE_SCALE_FACTOR)
    output = subprocess.check_output(command + coordinates)
    checkResult(output, UI_OUTPUT_RE, UI_NOISE_CHECK, "UI 40 with noise")

    # Clean up test file
    os.remove(INPUT_FILE_NAME)

def runMTFTest():

    # Formulate command line call for MTF
    dimensions = (CTP404_RADIUS_VX*2, CTP404_RADIUS_VX*2, CTP404_HEIGHT_VX)
    coordinates = [str(int(point)) for point in CTP404_CENTER_VX]
    command = [PYTHON, METRICS_APPLICATION, INPUT_FILE_NAME, METRICS_MTF_FLAG]

    # Generate infinite contrast dataset
    volume = np.zeros(dimensions, dtype=np.uint16)
    for (x, y, z), value in np.ndenumerate(volume):
        if (x-CTP404_RADIUS_VX)**2 + (y-CTP404_RADIUS_VX)**2 < CTP404_TEFLON_RADIUS_MM**2:
            volume[x][y][z] = CTP404_TEFLON_DENSITY
        else:
            volume[x][y][z] = CTP404_MODULE_DENSITY

    # Run IQ_Metrics script for infinite contrast dataset
    saveVolumeHdf5(INPUT_FILE_NAME, volume, dimensions, VOXEL_SIZE_MM, SAVE_SCALE_FACTOR)
    output = subprocess.check_output(command + coordinates)
    checkResult(output, MTF_OUTPUT_RE, MTF_IDEAL_CHECK, "MTF maximum contrast")

    # Generate 0.2 lp/mm contrast dataset
    beta = 0.5 # 50% MTF cutoff
    alpha = 2 * np.pi * 0.2 # lp/mm at 50% cutoff
    std = np.sqrt(-2 * np.log(beta) / alpha ** 2)
    for (x, y, z), value in np.ndenumerate(volume):
        r = np.sqrt((x-CTP404_RADIUS_VX)**2 + (y-CTP404_RADIUS_VX)**2)
        volume[x][y][z] = 1000 * st.norm.sf(r, CTP404_TEFLON_RADIUS_VX, std)

    # Run IQ_Metrics script for infinite contrast dataset
    saveVolumeHdf5(INPUT_FILE_NAME, volume, dimensions, VOXEL_SIZE_MM, SAVE_SCALE_FACTOR)
    output = subprocess.check_output(command + coordinates)
    checkResult(output, MTF_OUTPUT_RE, MTF_MEASURED_CHECK, "MTF 0.2 without noise")

    # Add noise to 0.2 lp/mm dataset
    volume = volume + np.random.normal(0, MTF_NOISE_STD, dimensions)

    # Run IQ_Metrics script for 0.2 lp/mm dataset with noise
    saveVolumeHdf5(INPUT_FILE_NAME, volume, dimensions, VOXEL_SIZE_MM, SAVE_SCALE_FACTOR)
    output = subprocess.check_output(command + coordinates)
    checkResult(output, MTF_OUTPUT_RE, MTF_NOISE_CHECK, "MTF 0.2 with noise")

    # Clean up test file
    os.remove(INPUT_FILE_NAME)

def runZSTTest():

    # Formulate parameters for ZST
    #dimensions = (ZST_WIDTH_VX, ZST_WIDTH_VX, ZST_HEIGHT_VX)
    dimensions = (ZST_HEIGHT_VX, ZST_HEIGHT_VX, ZST_HEIGHT_VX)
    pointShifts = np.linspace(-PLANE_RESOLUTION_MM/2,
                               PLANE_RESOLUTION_MM/2,
                               ZST_SLICE_COUNT)
    center = tuple(dimension/2 for dimension in dimensions)

    # Formulate command line call for ZST
    coordinates = [str(int(point)) for point in center]
    fileName = INPUT_FILE_BASE + "0".zfill(2) + INPUT_FILE_EXTENSION
    command = [PYTHON, METRICS_APPLICATION, fileName, METRICS_ZST_FLAG]

    # Generate volumes with 2.5 mm slice with sharp edges
    std = np.sqrt(-0.5 * (ZST_POINT_WIDTH/2.0)**2 / np.log(0.5))
    for index, pointShift in enumerate(pointShifts):

        # Start with background volume and iterate over volume dimensions
        volume = np.full(dimensions, ZST_BACKGROUND_DENSITY)
        x = np.arange(-ZST_HEIGHT_MM/2,
                       ZST_HEIGHT_MM/2,
                       SLICE_RESOLUTION_MM) - pointShift
        volume[center[0:2]] = ZST_POINT_DENSITY * st.norm.pdf(x, 0, std)

        # Output volume
        fileName = INPUT_FILE_BASE + str(index).zfill(2) + INPUT_FILE_EXTENSION
        saveVolumeHdf5(fileName, volume, dimensions, VOXEL_SIZE_MM, SAVE_SCALE_FACTOR)

    # Run ZST for 2.5 mm slice with rolloff edges
    output = subprocess.check_output(command + coordinates)
    checkResult(output, ZST_OUTPUT_RE, ZST_MEASURED_CHECK, "ZST 2.5mm without noise")

    # Generate volumes with 2.5 mm slice with gaussian edges and noise
    std = np.sqrt(-0.5 * (ZST_POINT_WIDTH/2.0)**2 / np.log(0.5))
    for index, pointShift in enumerate(pointShifts):

        # Start with background volume and iterate over volume dimensions
        volume = np.full(dimensions, ZST_BACKGROUND_DENSITY)
        x = np.arange(-ZST_HEIGHT_MM/2,
                       ZST_HEIGHT_MM/2,
                       SLICE_RESOLUTION_MM) - pointShift
        n = np.random.normal(0, ZST_NOISE_STD, len(x))
        volume[center[0:2]] = ZST_POINT_DENSITY * st.norm.pdf(x, 0, std) + n

        # Output volume
        fileName = INPUT_FILE_BASE + str(index).zfill(2) + INPUT_FILE_EXTENSION
        saveVolumeHdf5(fileName, volume, dimensions, VOXEL_SIZE_MM, SAVE_SCALE_FACTOR)

    # Run ZST for 2.5 mm slice with rolloff edges with noise
    output = subprocess.check_output(command + coordinates)
    checkResult(output, ZST_OUTPUT_RE, ZST_NOISE_CHECK, "ZST 2.5mm with noise")

def runNPSTest():

    # Formulate parameters for NPS
    dimensions = [CTP486_BACKGROUND_WIDTH_VX,
                  CTP486_BACKGROUND_WIDTH_VX,
                  CTP486_BACKGROUND_HEIGHT_VX]
    x = np.linspace(-CTP486_CENTER_MM[0], CTP486_CENTER_MM[0], dimensions[0])
    y = np.linspace(-CTP486_CENTER_MM[1], CTP486_CENTER_MM[1], dimensions[1])
    z = np.linspace(-CTP486_CENTER_MM[2], CTP486_CENTER_MM[2], dimensions[2])
    gridXY = np.sqrt(x[:,None]**2 + y[None,:]**2)
    whiteNoise = lambda: np.random.normal(0, NPS_NOISE_STD, dimensions)

    # Formulate command line call for NPS
    coordinates = [str(int(point)) for point in CTP486_CENTER_VX]
    fileName = INPUT_FILE_BASE + "0".zfill(2) + INPUT_FILE_EXTENSION
    command2d = [PYTHON, METRICS_APPLICATION, fileName, METRICS_NPS_FLAG]
    command3d = [PYTHON, METRICS_APPLICATION, fileName, METRICS_NPS3D_FLAG]

    # Generate and output data for white noise NPS test
    volume = np.full(dimensions, CTP486_MODULE_DENSITY)
    for index in range(NPS_SAMPLE_COUNT):
        fileName = INPUT_FILE_BASE + str(index).zfill(2) + INPUT_FILE_EXTENSION
        saveVolumeHdf5(fileName, volume*whiteNoise(), dimensions, VOXEL_SIZE_MM, SAVE_SCALE_FACTOR)

    # Run IQ_Metrics script for white noise
    #os.system(' '.join(command2d + coordinates))
    os.system(' '.join(command3d + coordinates))

    # Generate data for sine NPS test
    patternXY = np.sin(2*np.pi* NPS_SINE_FREQ * gridXY)
    planeXY = CTP486_MODULE_DENSITY/10 * patternXY + CTP486_MODULE_DENSITY
    volume = np.repeat(planeXY[:,:,None], dimensions[2], axis=2)
    for index in range(NPS_SAMPLE_COUNT):
        fileName = INPUT_FILE_BASE + str(index).zfill(2) + INPUT_FILE_EXTENSION
        saveVolumeHdf5(fileName, volume+whiteNoise(), dimensions, VOXEL_SIZE_MM, SAVE_SCALE_FACTOR)

    # Run IQ_Metrics script for NPS on sinen data with white noise
    #os.system(' '.join(command2d + coordinates))
    os.system(' '.join(command3d + coordinates))

    # Generate data for sinc NPS test
    planeXY = CTP486_MODULE_DENSITY/5 * np.sinc(gridXY/7) + CTP486_MODULE_DENSITY 
    sliceZ = CTP486_MODULE_DENSITY/10 * np.sin(2*np.pi* NPS_SINE_FREQ * z)
    volume = planeXY[:,:,None] + sliceZ[None,None,:]
    for index in range(NPS_SAMPLE_COUNT):
        fileName = INPUT_FILE_BASE + str(index).zfill(2) + INPUT_FILE_EXTENSION
        saveVolumeHdf5(fileName, volume+whiteNoise(), dimensions, VOXEL_SIZE_MM, SAVE_SCALE_FACTOR)

    # Run IQ_Metrics script for NPS on sinc function with white noise
    #os.system(' '.join(command2d + coordinates))
    os.system(' '.join(command3d + coordinates))

runCNRTest()
runUITest()
runMTFTest()
runZSTTest()
#runNPSTest()
