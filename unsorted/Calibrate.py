import numpy as np
import errno
import matplotlib
# We need to use Agg in the back instead of TK gui toolkit because it isn't threadsafe
# An brief explanation of this is found at this link:
# https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import h5py
import time
import subprocess as spr
import sys
import os
from datetime import datetime
from datetime import timedelta
from paramiko import SSHClient
import argparse

# ====== CONSTANTS =====================================================================

# Acquisition machine login
ACQUISITION_HOST = '169.254.0.1'
ACQUISITION_USER = 'nview'
ACQUISITION_PASSWORD = 'MedicalImaging'

# various command strings
ACQUISITION_DEFAULT_FILE_NAME = "DefaultStudy"
INITIAL_FILE_TAG = ".i."
VOLUME_EXTENSION = ".h5"
OFFSET_FILE_EXTENSION = ".StatsImage"
INVALID_FILE_PATH_CHARACTERS = '~`!@#$%^&*()+=[]{}\\|;:\'\",<>/?\ -._'
TEMP_FOLDER = "/tmp"
MAKE_DIRECTORY_COMMAND = "mkdir"
INSTALLATION_FOLDER = "/opt/rt3d/etc/"
CALIBRATION_FOLDER = "/opt/Calibration"
DARK_IMAGE_FILE_NAME = "darkImg.h5"
TECHNIQUE_COMP_FILE_NAME = "techniqueCompensation.txt"
SIA_MAP_FILE_NAME = "signalInAirMap.h5"
PM_FILE_NAME = "PM.cal"
TECHNIQUE_CALIBRATION_COMMAND = "python3 /opt/rt3d/bin/TechniqueCalibration.py"
LOG_FILE_NAME = "calibration.log" 
WHITE_IMAGE_FILE_NAME = "WhiteImageVerification_"
MKDIR_COMMAND = "mkdir"
TEST_COMMAND = "test -w"
SYM_LINK_COMMAND = "ln -s"
REMOVE_COMMAND = "rm"
PERIOD_CHAR = '.'

PRINT_VERBOSE = 1

# These are for reading in h5 files
VOXELDATA_PATH = "ITKImage/0/VoxelData"

# Image corrector constants
IMAGE_CORRECTOR_COMMAND = "ImageCorrector"
IMAGE_CORRECTOR_OFFSET_FLAG = "--offsetCorrection"
IMAGE_CORRECTOR_TECHNIQUE_FLAG = "--techniqueCorrection"
IMAGE_CORRECTOR_AIR_FLAG = "--airCorrection"
IMAGE_CORRECTOR_INPUT_FLAG = "--input"
IMAGE_CORRECTOR_OUTPUT_FLAG = "--output"
HIGH_DOSE_KV = "highDose"
MEDIUM_DOSE_KV = "mediumDose"
LOW_DOSE_KV = "lowDose"

# Motion command constants
MOTION_CALIBRATION_COMMAND = "MotionCalibration"
MOTION_CALIBRATION_INPUT_FLAG_STRING = "-i"
MOTION_CALIBRATION_VOLUME_FLAG_STRING = "-v"
MOTION_CALIBRATION_PM_FLAG = "-p"
MOTION_CALIBRATION_OUTPUT_FLAG_STRING = "--outputDirectory"

WHITE_IMAGE_VARIANCE_THRESHOLD = 2.25
SPREADSHEET_LINK = "https://docs.google.com/spreadsheets/d/1fWbaTdrw0QQQqXV7Olyk9JFgbPNTd60Ax2SziS6ekBo/edit#gid=0"

# Folder names
OFFSET_FOLDER_NAME = "_Offset"
TECHNIQUE_FOLDER_NAME = "_Technique"
SIA_FOLDER_NAME = "_SignalInAir"
VERIFICATION_FOLDER_NAME = "Verification"
MOTION_FOLDER_NAME = "_Geometric"
SEED_FOLDER_NAME = "Seed"

# Cooldown times
OFFSET_COOLDOWN_SECONDS = 0
TECHNIQUE_COOLDOWN_SECONDS = 300
SIA_COOLDOWN_SECONDS = 480
WHITE_IMAGE_COOLDOWN_TIME = 15
MOTION_COOLDOWN_SECONDS = 120
ACQUISITION_STARTUP_TIME = 25

# Messages to display before acquiring
OFFSET_ACQUIRE_MESSAGE = "Acquiring offset image. Please verify that nothing is in the path of the beam."
TECHNIQUE_ACQUIRE_MESSAGE = "Acquiring technique images. Please verify that nothing is in the path of the beam."
SIGNAL_IN_AIR_ACQUIRE_MESSAGE = "Acquiring signal in air image. Please verify that nothing is in the path of the beam."
WHITE_IMAGE_ACQUIRE_MESSAGE = "Acquiring white image. Please verify that nothing is in the path of the beam."
MOTION_ACQUIRE_MESSAGE = "Place the motion calibration phantom in the center of the beam.\n The phantom should be centered in the panel with the printed label facing away\n from the c-arm. The top of the phantom should be 1 cm away from the bottom\n of the panel."

# Commands to run acquisition
OFFSET_ACQUISITION_COMMAND = "Acquisition --quitAfter 1 --stats --writeToTCP false --FileIO.SaveAcquisition=1 --LowDose.kV=0 --LowDose.mA=0 --defaultDoseMode Low --saveDirectory"

TECHNIQUE_ACQUISITION_LOW_DOSE_COMMAND = "Acquisition --quitAfter 5 --writeToTCP false --FileIO.SaveAcquisition=1 --HighResAcquisition.MotorSpeed=0 --FluoroAcquisition.MotorSpeed=0 --motorZeroPosition --defaultDoseMode Low --saveDirectory"
TECHNIQUE_ACQUISITION_MEDIUM_DOSE_COMMAND = "Acquisition --quitAfter 5 --writeToTCP false --FileIO.SaveAcquisition=1 --HighResAcquisition.MotorSpeed=0 --FluoroAcquisition.MotorSpeed=0 --motorZeroPosition --defaultDoseMode Medium --saveDirectory"
TECHNIQUE_ACQUISITION_HIGH_DOSE_COMMAND = "Acquisition --quitAfter 5 --writeToTCP false --FileIO.SaveAcquisition=1 --HighResAcquisition.MotorSpeed=0 --FluoroAcquisition.MotorSpeed=0 --motorZeroPosition --defaultDoseMode High --saveDirectory"

SIA_ACQUISITION_LOW_DOSE_COMMAND = "Acquisition --quitAfter 1 --writeToTCP false --FileIO.SaveAcquisition=1 --HighResAcquisition.NumberOfViews=1000 --defaultDoseMode Low --saveDirectory"
SIA_ACQUISITION_MEDIUM_DOSE_COMMAND = "Acquisition --quitAfter 1 --writeToTCP false --FileIO.SaveAcquisition=1 --HighResAcquisition.NumberOfViews=1000 --defaultDoseMode Medium --saveDirectory"
SIA_ACQUISITION_HIGH_DOSE_COMMAND = "Acquisition --quitAfter 1 --writeToTCP false --FileIO.SaveAcquisition=1 --HighResAcquisition.NumberOfViews=1000 --defaultDoseMode High --saveDirectory"

WHITE_IMAGE_ACQUISITION_LOW_DOSE_COMMAND = "Acquisition --quitAfter 1 --FileIO.SaveAcquisition=1 --writeToTCP false --defaultDoseMode Low --saveDirectory"
WHITE_IMAGE_ACQUISITION_MEDIUM_DOSE_COMMAND = "Acquisition --quitAfter 1 --FileIO.SaveAcquisition=1 --writeToTCP false --defaultDoseMode Medium --saveDirectory"
WHITE_IMAGE_ACQUISITION_HIGH_DOSE_COMMAND = "Acquisition --quitAfter 1 --FileIO.SaveAcquisition=1 --writeToTCP false --defaultDoseMode High --saveDirectory"

MOTION_ACQUISITION_COMMAND = "Acquisition --quitAfter 1 --FileIO.SaveAcquisition=true --HighResAcquisition.NumberOfViews=500 --HighResAcquisition.MotorSpeed=90 --defaultDoseMode Medium --saveDirectory"

# ====== Globals =======================================================================
_nextAcquireTime = datetime.now()

# ====== Functions =====================================================================
def print_help():
    print("Use: python3 Calibrate.py outputFolder [offset technique signalInAir motion]")
    print("\t outputFolder:       Path to the base directory for this calibration")
    print("\t --offset:           Use this option to run the offset calibration")
    print("\t --technique:        Use this option to run the technique calibration")
    print("\t --signalInAir:      Use this option to run the signal in air calibration")
    print("\t --motion:           Use this option to run the motion calibration")
    print("\t --motionSeed:       Specify a path to a seed PM.cal file to use in motion calibration")
    print("\t NOTE: If none of the calibration options are specified, all calibrations will be run")

def run_offset_calibration(offsetOutDir):
    

    log("\nBeginning offset calibration.",PRINT_VERBOSE)

    # Set up the acquisition side directories
    # the folder on acquisition will be a unique name based on current timestamp
    acquisitionWorkingDirectory = os.path.join(TEMP_FOLDER, os.path.basename(offsetOutDir))

    # acquire the needed images
    acquireCommand = ' '.join([OFFSET_ACQUISITION_COMMAND, acquisitionWorkingDirectory])
    acquire(acquireCommand, OFFSET_COOLDOWN_SECONDS, OFFSET_ACQUIRE_MESSAGE)

    # move the files
    index = "0"
    acquisitionFileName = ACQUISITION_DEFAULT_FILE_NAME + INITIAL_FILE_TAG + index + OFFSET_FILE_EXTENSION + VOLUME_EXTENSION
    acquisitionFilePath = os.path.join(acquisitionWorkingDirectory, acquisitionFileName)
    darkImageFilePath = os.path.join(offsetOutDir, DARK_IMAGE_FILE_NAME)
    remote_copy_get(acquisitionFilePath, darkImageFilePath)

    # Create the symbolic link
    log("Installing darkImg.h5...")
    create_sym_link(darkImageFilePath, os.path.join(INSTALLATION_FOLDER,DARK_IMAGE_FILE_NAME))
    log("Offset calibration complete.",PRINT_VERBOSE)

def run_technique_calibration(techniqueOutDir):


    log("\nBeginning technique calibration.",PRINT_VERBOSE)

    # Set up the acquisition side directories
    acquisitionWorkingDirectory = os.path.join(TEMP_FOLDER, os.path.basename(techniqueOutDir))

    # acquire the needed images
    # low dose
    acquireCommand = ' '.join([TECHNIQUE_ACQUISITION_LOW_DOSE_COMMAND, acquisitionWorkingDirectory])
    acquire(acquireCommand, TECHNIQUE_COOLDOWN_SECONDS, TECHNIQUE_ACQUIRE_MESSAGE)

    # medium dose
    acquireCommand = ' '.join([TECHNIQUE_ACQUISITION_MEDIUM_DOSE_COMMAND, acquisitionWorkingDirectory])
    acquire(acquireCommand, TECHNIQUE_COOLDOWN_SECONDS, TECHNIQUE_ACQUIRE_MESSAGE)

    # high dose
    acquireCommand = ' '.join([TECHNIQUE_ACQUISITION_HIGH_DOSE_COMMAND, acquisitionWorkingDirectory])
    acquire(acquireCommand, TECHNIQUE_COOLDOWN_SECONDS, TECHNIQUE_ACQUIRE_MESSAGE)

    # move the files
    for index in range(0,15):
        print("copying file " +str(index+1) + "/15")
        acquisitionFileName = ACQUISITION_DEFAULT_FILE_NAME + INITIAL_FILE_TAG + str(index) + VOLUME_EXTENSION
        acquisitionFilePath = os.path.join(acquisitionWorkingDirectory, acquisitionFileName)
        remote_copy_get(acquisitionFilePath, os.path.join(techniqueOutDir,acquisitionFileName))

    # perform calibration

    print("Running the calibration with the acquired files...")
    techniqueCalibrationCommand = " ".join([TECHNIQUE_CALIBRATION_COMMAND, os.path.join(techniqueOutDir,'*'), os.path.join(techniqueOutDir,TECHNIQUE_COMP_FILE_NAME)])
    print(techniqueCalibrationCommand)

    #block until the calibration is done.
    returnCode = spr.Popen(techniqueCalibrationCommand, shell=True, stdout=spr.PIPE, stderr=spr.PIPE).wait()
    if ( returnCode != 0):
       log("Failed to complete command: " + techniqueCalibrationCommand, PRINT_VERBOSE)
       end_program()

    techniqueCompensationFilePath = os.path.join(techniqueOutDir, TECHNIQUE_COMP_FILE_NAME)

    # Install the link
    log("Installing " + TECHNIQUE_COMP_FILE_NAME + "...")
    create_sym_link(techniqueCompensationFilePath, os.path.join(INSTALLATION_FOLDER, TECHNIQUE_COMP_FILE_NAME))
    log("Technique calibration complete.",PRINT_VERBOSE)

def run_signal_in_air_calibration(signalInAirOutDir):

    log("\nBeginning signal in air calibration.",PRINT_VERBOSE)

    # Set up the acquisition side directories
    acquisitionWorkingDirectory = os.path.join(TEMP_FOLDER, os.path.basename(signalInAirOutDir))
    whiteHighResAcquisitionDirectory = os.path.join(acquisitionWorkingDirectory, VERIFICATION_FOLDER_NAME)
    whiteImageDirectory = create_directory(signalInAirOutDir, VERIFICATION_FOLDER_NAME)

    # acquire the needed images
    # low dose
    acquireCommand = ' '.join([SIA_ACQUISITION_LOW_DOSE_COMMAND, acquisitionWorkingDirectory])
    acquire(acquireCommand, SIA_COOLDOWN_SECONDS, SIGNAL_IN_AIR_ACQUIRE_MESSAGE)

    # medium dose
    acquireCommand = ' '.join([SIA_ACQUISITION_MEDIUM_DOSE_COMMAND, acquisitionWorkingDirectory])
    acquire(acquireCommand, SIA_COOLDOWN_SECONDS, SIGNAL_IN_AIR_ACQUIRE_MESSAGE)

    # high dose
    acquireCommand = ' '.join([SIA_ACQUISITION_HIGH_DOSE_COMMAND, acquisitionWorkingDirectory])
    acquire(acquireCommand, SIA_COOLDOWN_SECONDS, SIGNAL_IN_AIR_ACQUIRE_MESSAGE)

    # move the files
    for index in range(0,3):
        acquisitionFileName = ACQUISITION_DEFAULT_FILE_NAME + INITIAL_FILE_TAG + str(index) + VOLUME_EXTENSION
        acquisitionFilePath = os.path.join(acquisitionWorkingDirectory, acquisitionFileName)
        remote_copy_get(acquisitionFilePath, os.path.join(signalInAirOutDir,acquisitionFileName))

    # perform calibration
    signalInAirMapPath = os.path.join(signalInAirOutDir, SIA_MAP_FILE_NAME)
    siaCalCommand = ["SignalInAirCalibration", "-i", signalInAirOutDir, "--output", signalInAirMapPath, "--offsetCorrection", "--techniqueCorrection", "--medianFilter", "3", "--deadPix"]
    try:
        returnCode = spr.Popen(' '.join(siaCalCommand), shell=True).wait()
        if( returnCode != 0):
            log("Failed to complete command: " + ' '.join(siaCalCommand), PRINT_VERBOSE)
            end_program()

    except Exception as e:
        log("ERROR: could not complete command: " + siaCalCommand, PRINT_VERBOSE)
        log(str(e), PRINT_VERBOSE)
        end_program()

    # Install the link
    log("Installing " + SIA_MAP_FILE_NAME + "...")
    create_sym_link(signalInAirMapPath, os.path.join(INSTALLATION_FOLDER, SIA_MAP_FILE_NAME))

    # Perform white image verification
    log("Performing white image verification")

    # Low dose
    whiteImageAcquireCommand = ' '.join([WHITE_IMAGE_ACQUISITION_LOW_DOSE_COMMAND, whiteHighResAcquisitionDirectory])
    acquire(whiteImageAcquireCommand, WHITE_IMAGE_COOLDOWN_TIME, WHITE_IMAGE_ACQUIRE_MESSAGE)
    acquisitionFileName = ACQUISITION_DEFAULT_FILE_NAME + INITIAL_FILE_TAG + "0" + VOLUME_EXTENSION
    acquisitionFilePath = os.path.join(whiteHighResAcquisitionDirectory, acquisitionFileName)
    remote_copy_get(acquisitionFilePath, os.path.join(whiteImageDirectory, acquisitionFileName))

    uncorrectedFilePath = os.path.join(whiteImageDirectory, acquisitionFileName)
    correctedFilePath = os.path.join(whiteImageDirectory,WHITE_IMAGE_FILE_NAME + LOW_DOSE_KV + VOLUME_EXTENSION)
    corrections = [IMAGE_CORRECTOR_OFFSET_FLAG, IMAGE_CORRECTOR_TECHNIQUE_FLAG, IMAGE_CORRECTOR_AIR_FLAG]
    correct_image(uncorrectedFilePath, correctedFilePath, corrections)
    correctedVolume = get_image(correctedFilePath)

    stddev = np.std(correctedVolume)
    mean = np.mean(correctedVolume)
    percentVariance = 100 * stddev / mean
    log("White image percent variance: " + str(percentVariance),PRINT_VERBOSE)
    log("Please record this value in the calibration log spreadsheet.",PRINT_VERBOSE)
    log(SPREADSHEET_LINK,PRINT_VERBOSE)

    # Make a sound to prompt user
    # NOTE: display3d function causes the script to crash right now. Disable this capability 
    #print('\a')
    #input("Scan over the following image to verify image quality. Press enter when ready to continue...")
    #display3d(correctedVolume)

    if(percentVariance > WHITE_IMAGE_VARIANCE_THRESHOLD):
        log("WARNING: Variance is higher than threshold.",PRINT_VERBOSE)

    # medium dose
    whiteImageAcquireCommand = ' '.join([WHITE_IMAGE_ACQUISITION_MEDIUM_DOSE_COMMAND, whiteHighResAcquisitionDirectory])
    acquire(whiteImageAcquireCommand, WHITE_IMAGE_COOLDOWN_TIME, WHITE_IMAGE_ACQUIRE_MESSAGE)
    acquisitionFileName = ACQUISITION_DEFAULT_FILE_NAME + INITIAL_FILE_TAG + "1" + VOLUME_EXTENSION
    acquisitionFilePath = os.path.join(whiteHighResAcquisitionDirectory, acquisitionFileName)
    remote_copy_get(acquisitionFilePath, os.path.join(whiteImageDirectory, acquisitionFileName))

    uncorrectedFilePath = os.path.join(whiteImageDirectory, acquisitionFileName)
    correctedFilePath = os.path.join(whiteImageDirectory,WHITE_IMAGE_FILE_NAME + MEDIUM_DOSE_KV + VOLUME_EXTENSION)
    corrections = [IMAGE_CORRECTOR_OFFSET_FLAG, IMAGE_CORRECTOR_TECHNIQUE_FLAG, IMAGE_CORRECTOR_AIR_FLAG]
    correct_image(uncorrectedFilePath, correctedFilePath, corrections)
    correctedVolume = get_image(correctedFilePath)

    stddev = np.std(correctedVolume)
    mean = np.mean(correctedVolume)
    percentVariance = 100 * stddev / mean
    log("White image percent variance: " + str(percentVariance),PRINT_VERBOSE)
    log("Please record this value in the calibration log spreadsheet.",PRINT_VERBOSE)
    log(SPREADSHEET_LINK,PRINT_VERBOSE)

    # Make a sound to prompt user
    # NOTE: display3d function causes the script to crash right now. Disable this capability 
    #print('\a')
    #input("Scan over the following image to verify image quality. Press enter when ready to continue...")
    #display3d(correctedVolume)

    if(percentVariance > WHITE_IMAGE_VARIANCE_THRESHOLD):
        log("WARNING: Variance is higher than threshold.",PRINT_VERBOSE)

    # high dose
    whiteImageAcquireCommand = ' '.join([WHITE_IMAGE_ACQUISITION_HIGH_DOSE_COMMAND, whiteHighResAcquisitionDirectory])
    acquire(whiteImageAcquireCommand, WHITE_IMAGE_COOLDOWN_TIME, WHITE_IMAGE_ACQUIRE_MESSAGE)
    acquisitionFileName = ACQUISITION_DEFAULT_FILE_NAME + INITIAL_FILE_TAG + "2" + VOLUME_EXTENSION
    acquisitionFilePath = os.path.join(whiteHighResAcquisitionDirectory, acquisitionFileName)
    remote_copy_get(acquisitionFilePath, os.path.join(whiteImageDirectory, acquisitionFileName))

    uncorrectedFilePath = os.path.join(whiteImageDirectory, acquisitionFileName)
    correctedFilePath = os.path.join(whiteImageDirectory,WHITE_IMAGE_FILE_NAME + HIGH_DOSE_KV + VOLUME_EXTENSION)
    corrections = [IMAGE_CORRECTOR_OFFSET_FLAG, IMAGE_CORRECTOR_TECHNIQUE_FLAG, IMAGE_CORRECTOR_AIR_FLAG]
    correct_image(uncorrectedFilePath, correctedFilePath, corrections)
    correctedVolume = get_image(correctedFilePath)

    stddev = np.std(correctedVolume)
    mean = np.mean(correctedVolume)
    percentVariance = 100 * stddev / mean
    log("White image percent variance: " + str(percentVariance),PRINT_VERBOSE)
    log("Please record this value in the calibration log spreadsheet.",PRINT_VERBOSE)
    log(SPREADSHEET_LINK,PRINT_VERBOSE)

    # Make a sound to prompt user
    # NOTE: display3d function causes the script to crash right now. Disable this capability 
    #print('\a')
    #input("Scan over the following image to verify image quality. Press enter when ready to continue...")
    #display3d(correctedVolume)

    if(percentVariance > WHITE_IMAGE_VARIANCE_THRESHOLD):
        log("WARNING: Variance is higher than threshold.",PRINT_VERBOSE)

    log("Signal in air calibration complete.",PRINT_VERBOSE)

def runMotionCalibration(motionOutDir, seedPath):

    log("\nBeginning motion calibration.",PRINT_VERBOSE)

    # Set up the acquisition side directories
    acquisitionWorkingDirectory = os.path.join(TEMP_FOLDER, os.path.basename(motionOutDir))

    # acquire the needed images
    acquireCommand = ' '.join([MOTION_ACQUISITION_COMMAND, acquisitionWorkingDirectory])
    acquire(acquireCommand, MOTION_COOLDOWN_SECONDS, MOTION_ACQUIRE_MESSAGE)

    # move the files
    index = "0"
    acquisitionFileName = ACQUISITION_DEFAULT_FILE_NAME + INITIAL_FILE_TAG + index + VOLUME_EXTENSION
    acquisitionFilePath = os.path.join(acquisitionWorkingDirectory, acquisitionFileName)
    motionAcqFilePath = os.path.join(motionOutDir, acquisitionFileName)
    remote_copy_get(acquisitionFilePath, motionAcqFilePath)

    # get the ct ground truth path
    ctPath = ""
    while(True):
        ctPath = input("Please enter the location of the CT ground truth file: ")
        if(os.path.isfile(ctPath) and ctPath.endswith('.h5')):
            break
        print("Invalid file path.")

    # Check if the user gave a seed. If they did, use that. Otherwise, create a seed.
    if( seedPath is None):
        log("Running motion calibration without a seed.", PRINT_VERBOSE)
        motionCalibrationCommand = [MOTION_CALIBRATION_COMMAND,
                                    MOTION_CALIBRATION_INPUT_FLAG_STRING, motionAcqFilePath, 
                                    MOTION_CALIBRATION_VOLUME_FLAG_STRING, ctPath, 
                                    MOTION_CALIBRATION_OUTPUT_FLAG_STRING, motionOutDir]
        try:
            returnCode = spr.Popen(motionCalibrationCommand).wait()
            if( returnCode != 0):
                log("Unable to complete command: " + motionCalibrationCommand)
                end_program()
        except Exception as e:
            log("Unable to complete the command " + ' '.join(motionCalibrationCommand), PRINT_VERBOSE)
            log(str(e))
            end_program()
    else:
        log("Running motion calibration.")
        # Run the calibration
        motionCalibrationCommand = [MOTION_CALIBRATION_COMMAND, 
                                    MOTION_CALIBRATION_INPUT_FLAG_STRING, motionAcqFilePath, 
                                    MOTION_CALIBRATION_VOLUME_FLAG_STRING, ctPath, 
                                    MOTION_CALIBRATION_OUTPUT_FLAG_STRING, motionOutDir, 
                                    MOTION_CALIBRATION_PM_FLAG, seedPath]
        try:
            returnCode = spr.Popen(motionCalibrationCommand).wait()
            if( returnCode != 0):
                log("Unable to complete command: " + motionCalibrationCommand, PRINT_VERBOSE)
                end_program()
        except Exception as e:
            log("Unable to complete the command " + ' '.join(motionCalibrationCommand), PRINT_VERBOSE)
            log(str(e))
            end_program()

    # Instructions to manually check the difference images
    log("Before continuing, please check the error projection post calibration image.", PRINT_VERBOSE)
    log("The black BBs and the white BBs should be aligned with each other." , PRINT_VERBOSE)
    input("Press enter when this is done and you are ready to continue...\n")

    # Install the link to the acquisition PC
    log("Installing " + PM_FILE_NAME + "...", PRINT_VERBOSE)

    # Create the remote directory
    now = datetime.now()
    remoteFolderName = os.path.basename(os.path.normpath(motionOutDir))
    create_remote_directory(CALIBRATION_FOLDER, remoteFolderName)
    remoteFolderName = os.path.join(CALIBRATION_FOLDER, remoteFolderName)

    # Send the file to the new folder
    pmCalPath = os.path.join(motionOutDir, PM_FILE_NAME)
    remote_copy_put(pmCalPath, os.path.join(remoteFolderName, PM_FILE_NAME))

    # Add a link to the install folder
    remove_remote_file(os.path.join(INSTALLATION_FOLDER, PM_FILE_NAME))
    create_remote_sym_link(os.path.join(remoteFolderName, PM_FILE_NAME), os.path.join(INSTALLATION_FOLDER, PM_FILE_NAME))

    log("Motion calibration complete.",PRINT_VERBOSE)

def get_image(path):
    with h5py.File(path, 'r') as imgFile:
        return imgFile[VOXELDATA_PATH][:]

# NOTE: display3d function causes the script to crash right now. Disable this capability 
def display3d(img, axis=0, **kwargs):
    # check dim
    if not img.ndim == 3:
        raise ValueError("img should be an ndarray with ndim == 3")

    # generate figure
    fig = plt.figure()
    ax = plt.subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # select first image
    s = [slice(0, 1) if i == axis else slice(None) for i in xrange(3)]
    im = img[tuple(s)].squeeze()

    # display image
    l = ax.imshow(im, **kwargs)

    plt.show()

def correct_image(inPath, outPath, corrections):
    cmd =  [IMAGE_CORRECTOR_COMMAND, IMAGE_CORRECTOR_INPUT_FLAG, inPath, IMAGE_CORRECTOR_OUTPUT_FLAG, outPath] + corrections
    print(' '.join(cmd))
    try:
        if( spr.Popen(cmd).wait() != 0):
            log("Unable to complete command: " + ' '.join(cmd), PRINT_VERBOSE)
            end_program()
    except Exception as e:
        log("Unable to complete command: " + " ".join(cmd),PRINT_VERBOSE)
        log(str(e),PRINT_VERBOSE)

def acquire(acquireCommand, cooldownTime, message):

    global _nextAcquireTime
    global acquisitionClient

    block_until_xray_ready()
    log("Preparing to take acquisitions.")

    log(message, PRINT_VERBOSE)

    # Make a sound
    print('\a')
    # Block until the user presses enter
    input("\nPress enter when this is done and you are ready to shoot x-rays.")
    print("Acquiring...")

    # calculate next acquire time
    _nextAcquireTime = datetime.now() + timedelta(seconds=(cooldownTime + ACQUISITION_STARTUP_TIME))
    try:
        sin,sout,serr = acquisitionClient.exec_command(acquireCommand)
        # block until done
        exitStatus = sout.channel.recv_exit_status
        if(exitStatus() != 0):
            log("Error completing remote command: " + acquireCommand,PRINT_VERBOSE)
            end_program()
            return
    except Exception as e:
        print("Failed to complete remote command: " + acquireCommand,PRINT_VERBOSE)
        print(str(e))
        end_program()

def open_log_file(saveFolder):

    global logFile
    logFilePath = os.path.join(saveFolder, LOG_FILE_NAME)
    # the file exists already, so remove it
    if os.path.isfile(logFilePath):
        os.remove(logFilePath)

    try:
        print("Log file saving at: " + logFilePath)
        logFile = open(logFilePath, 'w')
    except Exception as e:
        print('Failed to open log file: ' +  str(e))
        print_help()
        exit()

def log(msg, verbosity=0):
    global logFile

    if(verbosity):
        print(msg)
    logFile.write(msg + '\n')

def create_directory(path, newfolder):
    outPath = os.path.join(path, newfolder)
    try:
        os.mkdir(outPath)
    except Exception as e:
        log("Unable to create directory: " + outPath + ", " + str(e),PRINT_VERBOSE)
    return outPath 

def create_remote_directory(path, newfolder):
    global acquisitionClient
    cmd = ' '.join([MKDIR_COMMAND, os.path.join(path, newfolder)]) 
    try:
        acquisitionClient.exec_command(cmd)
    except Exception as e:
        print ('Failed to create remote directory' + str(e))
        exit()

def move(path, destination):
    try:
        os.rename(path, destination)
    except:
        log("Unable to move: " + linkDest + ", " + str(e),PRINT_VERBOSE)
        end_program()

def check_remote_file_is_writable(target):
    global acquisitionClient
    cmd = ' '.join([TEST_COMMAND, target])
    try:
        acquisitionClient.exec_command(cmd)
        sin,sout,serr = acquisitionClient.exec_command(acquireCommand)
        return sout.channel.recv_exit_status()
    except Exception as e:
        print ('Unable to check remote file exists' + str(e))
        exit()

def create_sym_link(inFile, linkDest):
    try:
        os.symlink(inFile, linkDest)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(linkDest)
            create_sym_link(inFile,linkDest)
    except Exception as e:
        log("Unable to create a symlink at: " + linkDest + ", " + str(e),PRINT_VERBOSE)
        end_program()

def create_remote_sym_link(target, dest):

    # Check if the file already exists with sudo permissions. 
     try:
        print(sftp.stat('/tmp/deleteme.txt'))
        print('file exists')
    except IOError:
        print('copying file')
        sftp.put('deleteme.txt', '/tmp/deleteme.txt')
    print("ERROR: Failed to create remote symlink at", dest)
    print("Please remove the file on the c-arm at", dest, "and create a symbolic link using the following commands on the acquisition PC:")
    print("sudo rm",dest)
    print(cmd)
    print("NOTE: Do not use sudo on the previous command. Doing so will cause the automatic PM.cal install to fail in the future.")
    global acquisitionClient
    cmd = ' '.join([SYM_LINK_COMMAND, target, dest])
    try:
        acquisitionClient.exec_command(cmd)
    except Exception as e:
        print ('Failed to create remote symlink' + str(e))
        exit()

def remove_remote_file(target):
    global acquisitionClient
    cmd = ' '.join([REMOVE_COMMAND, target])
    try:
        acquisitionClient.exec_command(cmd)
    except Exception as e:
        print ('Failed to remove remote file' + str(e))
        exit()

def connect_acquisition_client():

    log("Connecting to " + ACQUISITION_HOST, PRINT_VERBOSE)
    try:
        client = SSHClient()
        client.load_system_host_keys()
        client.connect(ACQUISITION_HOST, username=ACQUISITION_USER, password=ACQUISITION_PASSWORD)
    except Exception as e:
        log('Failed to connect to Acquisition PC '+  str(e),PRINT_VERBOSE)
        exit()

    return client

def remote_copy_get(target, destination):
    global acquisitionClient
    try:
        client = acquisitionClient.open_sftp()
        client.get(target,destination)
        client.close()
    except Exception as e:
        log("Unable to remotely copy file: " + target + ", "  + str(e),PRINT_VERBOSE)
        end_program()

def remote_copy_put(target, dest):
    global acquisitionClient
    try:
        client = acquisitionClient.open_sftp()
        client.put(target,dest)
        client.close()
    except Exception as e:
        log("Unable to remotely copy file: " + target + ", "  + str(e),PRINT_VERBOSE)
        end_program()

def block_until_xray_ready():
    global _nextAcquireTime
    log("Blocking until tube is sufficiently cool. Next acquire time: " + str(_nextAcquireTime) ,PRINT_VERBOSE)
    while datetime.now() < _nextAcquireTime:
        time.sleep(5)
    log("Done blocking.",PRINT_VERBOSE)

def parse_inputs(args):

    parser = argparse.ArgumentParser()
    parser.add_argument("outputDirectory", help="The base directory for the outputs of this calibration")
    parser.add_argument("-o","--offset", help="If this flag is present, offset calibration will be run", action="store_true")
    parser.add_argument("-t","--technique", help="If this flag is present, technique calibration will be run", action="store_true")
    parser.add_argument("-a","--signalInAir", help="If this flag is present, signal in air calibration will be run", action="store_true")
    parser.add_argument("-m","--motion", help="If this flag is present, motion calibration will be run", action="store_true")
    parser.add_argument("-s","--motionSeed", help="Run the motion calibration only once using the specified seed")

    parsedList = parser.parse_args(args).__dict__

    # change relative paths to absolute paths
    outputFolder = os.path.abspath(parsedList["outputDirectory"])

    if(not os.path.isdir(outputFolder)):
        print("Invalid argument for outputDirectory.")
        print_help()
        exit()
    del parsedList['outputDirectory']

    # if none are specified, set them all to true
    if(not any(parsedList[val] for val in parsedList)):
        parsedList['offset'] = True
        parsedList['technique'] = True
        parsedList['signalInAir'] = True
        parsedList['motion'] = True

    # if motion seed is specified, make sure the path exists
    if( parsedList['motionSeed'] is not None ):
        if( not os.path.isfile(parsedList['motionSeed'])):
            print("Invalid path for motion seed.")
            print_help()
            exit()

    return outputFolder, parsedList

def end_program():
    global acquisitionClient
    global _nextAcquireTime
    acquisitionClient.close()
    log("Ending program",PRINT_VERBOSE)
    log("Let tube cool before acquiring. Next acquire time: " + str(_nextAcquireTime),PRINT_VERBOSE)
    logFile.close()
    print("closed log file")
    exit()

# ====== Main ==========================================================================
def main():

    global acquisitionClient

    outputFolderPath, calibrationInputFlags = parse_inputs(sys.argv[1:])
    open_log_file(outputFolderPath)

    log("Running automatic calibration", PRINT_VERBOSE)

    acquisitionClient = connect_acquisition_client()
    
    invalidCharTable = str.maketrans(dict.fromkeys(INVALID_FILE_PATH_CHARACTERS))

    if(calibrationInputFlags.get("offset")):
        offsetOutDir = create_directory(outputFolderPath, str(datetime.now()).translate(invalidCharTable) + OFFSET_FOLDER_NAME)
        run_offset_calibration(offsetOutDir)
    if(calibrationInputFlags.get("technique")):
        techniqueOutDir = create_directory(outputFolderPath, str(datetime.now()).translate(invalidCharTable) + TECHNIQUE_FOLDER_NAME)
        run_technique_calibration(techniqueOutDir)
    if(calibrationInputFlags.get("signalInAir")):
        signalInAirOutDir = create_directory(outputFolderPath, str(datetime.now()).translate(invalidCharTable) + SIA_FOLDER_NAME)
        run_signal_in_air_calibration(signalInAirOutDir)
    if(calibrationInputFlags.get("motion")):
        motionOutDir = create_directory(outputFolderPath, str(datetime.now()).translate(invalidCharTable) + MOTION_FOLDER_NAME)
        run_motion_calibration(motionOutDir, calibrationInputFlags['motionSeed'])

if __name__ == "__main__":
    main()
