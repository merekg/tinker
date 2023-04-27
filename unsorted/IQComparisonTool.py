import subprocess as spr
from paramiko import SSHClient
import skimage.measure as skm
import numpy as np
import h5py
import sys
import csv
import os
import re

# Keys for reading and writing CSV file
CSV_FILE_KEYS = [
        'Comparison Name',
        'Playback Flags',
        'Reconstruction Flags',
        'Save Path',
        'Final L1/N',
        'Layer 0 Iterations',
        'Layer 0 Time',
        'Layer 0 L1/N',
        'Layer 1 Iterations',
        'Layer 1 Time',
        'Layer 1 L1/N',
        'Comparison Path',
        'Comparison L1/N',
        'Comparison MSE',
        'Comparison Correlation',
        'Comparison SSIM',
        'Pass/Fail',
        'Fail Criteria']

# Acquisition machine login
ACQUISITION_HOST = '169.254.0.1'
ACQUISITION_USER = 'nview'
ACQUISITION_PASSWORD = 'MedicalImaging'

# Commands and regex for finding reconstruction version number
SOFTWARE_VERSION_COMMAND = ['Reconstruction', '-h']
SOFTWARE_VERSION_REGEX = re.compile('Running Reconstruction version ([0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+-[a-z0-9]+)?)')

# Application paths
RECONSTRUCTION_APP = 'Reconstruction'
RECONSTRUCTION_SAVE_PATH_FLAG = '--overrideFilePath'
RECONSTRUCTION_SINGLE_FLAGS = ['--quitAfter1', '--FileIO.SaveReconstruction=true', '--FileIO.SaveAcquisition=false']
RECONSTRUCTION_DUAL_FLAGS = ['--quitAfterDual', '--FileIO.SaveReconstruction=false', '--FileIO.SaveAcquisition=false']
PLAYBACK_APP = 'AcquisitionPlayback'
PLAYBACK_SINGLE_FLAGS = ['--writeToDatabase=false', '-r', '169.254.0.2', '-i', '169.254.0.1', '--quitAfterPlaylist', '--autoPlayback', '10', '--autoDelay', '10']
PLAYBACK_DUAL_FLAGS = ['--writeToDatabase=false', '-r', '169.254.0.2', '-i', '169.254.0.1',  '--quitAfterPlaylist', '--autoPlayback', '10', '--autoDelay', '30']

# File save path consts
RECONSTRUCTION_FILE_EXTENSION = '.h5'
INVALID_FILE_PATH_CHARACTERS = '~`!@#$%^&*()+=[]{}\\|;:\'\",<>/?'
LOG_FILE_NAME = 'comparison.log'
REPORT_FILE_NAME = 'comparison.csv'

# HDF5 paths for reconstruction volumes
HDF5_VOLUME_PATH = 'ITKImage/0/VoxelData'
HDF5_SCALE_ATTRIBUTE = 'scale'

# Regexes for reading reconstruction metrics
LAYER0_L1ON_REGEX = re.compile('Recon time for layer 0: ~[0-9]+ ms, l1 over N: ([0-9.]+)')
LAYER0_TIME_REGEX = re.compile('Recon time for layer 0: ~([0-9]+) ms, l1 over N: [0-9.]+')
LAYER1_L1ON_REGEX = re.compile('Recon time for layer 1: [0-9]+ ms, l1 over N: ([0-9.]+)')
LAYER1_TIME_REGEX = re.compile('Recon time for layer 1: ([0-9]+) ms, l1 over N: [0-9.]+')

# Thresholds for reconstruction metrics
L10N_LAYER_THRESHOLD = 1.1
TIME_THRESHOLD_MS = 1500

# Thresholds for image comparisons
L1ON_COMPARISON_THRESHOLD = 0.0025
MSE_COMPARISON_THRESHOLD = 0.25
CORRELATION_COMPARISON_THRESHOLD = 10
SSIM_COMPARISON_THRESHOLD = 0.9

def printHelp():

    print 'Summary: This application will reconstruct and perform image quality comparisons of a set of acquisitions provided in a CSV file. The CSV file must be \'semicolon\' separated with a line for each IQ comparison. The columns of the file should follow the order', ';'.join(CSV_FILE_KEYS), '. The output will be given in the folder specified'
    print 'Usage: python IQComparisonTool.py <input-csv-file> <output-folder>'

def processArguments(argumentsList):

    try:
        assert len(argumentsList) == 3
        assert os.path.isfile(argumentsList[1])
        assert os.path.isdir(argumentsList[2])
    except Exception as e:
        print 'Invalid command line arguments', str(e)
        printHelp()
        exit()

    return argumentsList[1], argumentsList[2]

def getSoftwareVersion():

    try:
        versionProccess = spr.Popen(SOFTWARE_VERSION_COMMAND, stdout=spr.PIPE, stderr=spr.PIPE)
        versionOutPrint, versionErrPrint = versionProccess.communicate()
        versionCheck = SOFTWARE_VERSION_REGEX.search(versionErrPrint)
        softwareVersion = versionCheck.group(1)
    except Exception as e:
        print 'Failed to identify software version:', str(e)
        printHelp()
        exit()

    return softwareVersion

def createSaveFolder(outputFolder, softwareVersion):

    try:
        saveFolder = os.path.join(outputFolder, softwareVersion)
        os.mkdir(saveFolder)
    except Exception as e:
        print 'Failed to create output folder', str(e)
        printHelp()
        exit()

    return saveFolder

def loadComparisonList(inputCSVPath):

    try:
        reader = csv.reader(open(inputCSVPath), delimiter=';')
        comparisonList = [dict(zip(CSV_FILE_KEYS, row)) for row in list(reader)]
    except Exception as e:
        print 'Failed to load input CSV file', str(e)
        printHelp()
        exit()

    return comparisonList[1:]

def openLogFile(saveFolder):

    global logFile

    try:
        logFilePath = os.path.join(saveFolder, LOG_FILE_NAME)
        logFile = open(logFilePath, 'w')
    except Exception as e:
        print 'Failed to open log file', str(e)
        printHelp()
        exit()

def openReportFile(saveFolder):

    try:
        reportFilePath = os.path.join(saveFolder, REPORT_FILE_NAME)
        reportFile = open(reportFilePath, 'w')
        reportFile.write(';'.join(CSV_FILE_KEYS))
        reportFile.write('\n')
    except Exception as e:
        print 'Failed to open report file', str(e)
        printHelp()
        exit()

    return reportFile

def connectAcquisitionClient():

    try:
        client = SSHClient()
        client.load_system_host_keys()
        client.connect(ACQUISITION_HOST, username=ACQUISITION_USER, password=ACQUISITION_PASSWORD)
    except Exception as e:
        print 'Failed to connect to Acquisition PC', str(e)
        printHelp()
        exit()

    return client

def getSavePath(saveFolder, comparison):

   fileName = '_'.join(comparison['Comparison Name'].split(' ')) + RECONSTRUCTION_FILE_EXTENSION 
   fileName = fileName.translate(None, INVALID_FILE_PATH_CHARACTERS)
   return os.path.abspath(os.path.join(saveFolder, fileName))

def getReconstructionCommand(comparison, reconstructionSavePath):

    providedFlags = comparison['Reconstruction Flags'].split(' ')
    savePathFlags = [RECONSTRUCTION_SAVE_PATH_FLAG, reconstructionSavePath]

    if '-p' in comparison['Playback Flags']:
        reconstructionFlags = providedFlags + savePathFlags + RECONSTRUCTION_DUAL_FLAGS
    else:
        reconstructionFlags = providedFlags + savePathFlags + RECONSTRUCTION_SINGLE_FLAGS

    return [RECONSTRUCTION_APP] + reconstructionFlags

def getPlaybackCommand(comparison):

    providedFlags = comparison['Playback Flags'].split(' ')

    if '-p' in comparison['Playback Flags']:
        playbackFlags = providedFlags + PLAYBACK_DUAL_FLAGS
    else:
        playbackFlags = providedFlags + PLAYBACK_SINGLE_FLAGS

    return ' '.join([PLAYBACK_APP] + playbackFlags)

def loadVolume(volumePath):

    global logFile

    try:
        with h5py.File(volumePath, 'r') as volumeFile:
            voxelData = volumeFile[HDF5_VOLUME_PATH][:].astype(float)
            scaleFactor = float(volumeFile[HDF5_VOLUME_PATH].attrs[HDF5_SCALE_ATTRIBUTE])
            volume = voxelData / scaleFactor
    except Exception as e:
        logFile.write('Failed to load volume: ' + volumePath + ' : ' + str(e) + '\n')
        volume = np.array([])

    return volume

def recordResult(reportFile, result):

    for key in CSV_FILE_KEYS[:-1]:
        reportFile.write(result[key] + ';')
    reportFile.write(result[CSV_FILE_KEYS[-1]] + '\n')

def compareL1ON(volume1, volume2):

    global logFile

    try:
        l1on = np.average(np.abs(volume1 - volume2))
    except Exception as e:
        logFile.write('Failed to compute L1/N comparison: ' + str(e) + '\n')
        l1on = np.inf

    return l1on

def compareMSE(volume1, volume2):

    global logFile

    try:
        mse = skm.compare_nrmse(volume1, volume2)
    except Exception as e:
        logFile.write('Failed to compute MSE comparison: ' + str(e) + '\n')
        mse = np.inf

    return mse

def compareCorrelation(volume1, volume2):

    global logFile

    try:
        correlation = np.sum((volume1 * volume2)**2)
        normalizeFactor = np.sqrt(np.sum(volume1**2) * np.sum(volume2**2))
        normalizedCorrelation = correlation / normalizeFactor
    except Exception as e:
        logFile.write('Failed to compute correlation comparison: ' + str(e) + '\n')
        normalizedCorrelation = np.inf

    return normalizedCorrelation

def compareSSIM(volume1, volume2):

    global logFile

    try:
        ssim = skm.compare_ssim(volume1, volume2)
    except Exception as e:
        logFile.write('Failed to compute SSIM comparison: ' + str(e) + '\n')
        ssim = np.inf

    return ssim

def getReconMetrics(reconOut):

    global logFile

    metrics = {
            'Final L1/N' : np.inf,
            'Layer 0 Iterations' : 0,
            'Layer 0 Time' : np.inf,
            'Layer 0 L1/N' : np.inf,
            'Layer 1 Iterations' : 0,
            'Layer 1 Time' : np.inf,
            'Layer 1 L1/N' : np.inf
            }

    try:
        for line in reconOut.split('\n'):
            if LAYER0_L1ON_REGEX.search(line):
                metrics['Layer 0 L1/N'] = float(LAYER0_L1ON_REGEX.search(line).group(1))
            if LAYER0_TIME_REGEX.search(line):
                metrics['Layer 0 Time'] = int(LAYER0_TIME_REGEX.search(line).group(1))
            if LAYER1_L1ON_REGEX.search(line):
                metrics['Layer 1 L1/N'] = float(LAYER1_L1ON_REGEX.search(line).group(1))
            if LAYER1_TIME_REGEX.search(line):
                metrics['Layer 1 Time'] = int(LAYER1_TIME_REGEX.search(line).group(1))
    except Exception as e:
        logFile.write('Failed to identify metrics in recon output: ' + str(e) + '\n')

    return metrics

def checkMetrics(metrics, comparison):

    check = True
    criteria = ''

    try:
        if metrics['Layer 0 L1/N'] > float(comparison['Layer 0 L1/N']) * L10N_LAYER_THRESHOLD:
            check = False
            criteria += 'Layer 0 L1/N exceeded by more than' + str(L10N_LAYER_THRESHOLD) + ' times'

        if metrics['Layer 1 L1/N'] > float(comparison['Layer 1 L1/N']) * L10N_LAYER_THRESHOLD:
            check = False
            criteria += 'Layer 1 L1/N exceeded by more than ' + str(L10N_LAYER_THRESHOLD) + ' times'

        if metrics['Layer 0 Time'] > int(comparison['Layer 0 Time']) + TIME_THRESHOLD_MS:
            check = False
            criteria += 'Layer 0 recon time increased by more than ' + str(TIME_THRESHOLD_MS) + ' ms'

        if metrics['Layer 1 Time'] > int(comparison['Layer 1 Time']) + TIME_THRESHOLD_MS:
            check = False
            criteria += 'Layer 1 recon time increased by more than ' + str(TIME_THRESHOLD_MS) + ' ms'
    except:
        failure = 'Failed to compare reconstruction metrics'
        check = False
        criteria += failure

    return check, criteria

def checkComparison(comparisonL1ON, comparisonMSE, comparisonCorrelation, comparisonSSIM):

    check = True
    criteria = ''

    if comparisonL1ON > L1ON_COMPARISON_THRESHOLD:
        check = False
        excess = comparisonL1ON - L1ON_COMPARISON_THRESHOLD
        criteria += 'L1/N comparison threshold exceeded by ' + str(excess)

    if comparisonMSE > MSE_COMPARISON_THRESHOLD:
        check = False
        excess = comparisonMSE - MSE_COMPARISON_THRESHOLD
        criteria += 'MSE comparison threshold exceeded by ' + str(excess)

    if comparisonCorrelation > CORRELATION_COMPARISON_THRESHOLD:
        check = False
        excess = comparisonCorrelation - CORRELATION_COMPARISON_THRESHOLD
        criteria += 'Correlation comparison threshold exceeded by ' + str(excess)

    if comparisonSSIM < SSIM_COMPARISON_THRESHOLD:
        check = False
        excess = SSIM_COMPARISON_THRESHOLD - comparisonSSIM
        criteria += 'SSIM comparison threshold exceeded by ' + str(excess)

    return check, criteria

if __name__ == '__main__':

    global logFile

    # Setup files and infrastructure
    inputFile, outputFolder = processArguments(sys.argv)
    softwareVersion = getSoftwareVersion()
    saveFolder = createSaveFolder(outputFolder, softwareVersion)
    openLogFile(saveFolder)
    reportFile = openReportFile(saveFolder)
    comparisonList = loadComparisonList(inputFile)
    acquisitionClient = connectAcquisitionClient()

    # For each comparison given in the list reconstruct, compare, and report
    for comparison in comparisonList:

        # Make save path of reconstruction
        reconstructionSavePath = getSavePath(saveFolder, comparison)

        # Get reconstruction and playback commands for comparison
        reconCommand = getReconstructionCommand(comparison, reconstructionSavePath)
        playbackCommand = getPlaybackCommand(comparison)

        # Launch Reconstruction
        logFile.write('Running reconstruction with command: ' + ' '.join(reconCommand) + '\n')
        reconProcess = spr.Popen(reconCommand, stdout=spr.PIPE, stdin=spr.PIPE, stderr=spr.PIPE)

        # Launch Playback
        logFile.write('Running playback with command: ' + playbackCommand + '\n')
        playbackIn, playbackOut, playbackErr = acquisitionClient.exec_command(playbackCommand)

        # Block until Reconstruction completes
        reconOut, reconErr = reconProcess.communicate()

        # Log console output from Playback and Reconstruction
        logFile.write(''.join(playbackErr.readlines()))
        logFile.write(''.join(playbackOut.readlines()))
        logFile.write(reconOut)
        logFile.write(reconErr)

        # Compare against previous reconstruction
        reconstructionVolume = loadVolume(reconstructionSavePath)
        comparisonVolume = loadVolume(comparison['Save Path'])

        # Read metrics from reconstruction output
        reconMetrics = getReconMetrics(reconOut)

        # Compute metrics against comparison volume
        comparisonL1ON = compareL1ON(reconstructionVolume, comparisonVolume)
        comparisonMSE = compareMSE(reconstructionVolume, comparisonVolume)
        comparisonCorrelation = compareCorrelation(reconstructionVolume, comparisonVolume)
        comparisonSSIM = compareSSIM(reconstructionVolume, comparisonVolume)

        # Check metrics
        metricsPass, metricsCriteria = checkMetrics(reconMetrics, comparison)
        comparisonPass, comparisonCriteria = checkComparison(comparisonL1ON, comparisonMSE, comparisonCorrelation, comparisonSSIM)
        comparisonResult = 'PASS' if metricsPass and comparisonPass else 'FAIL'

        # Record and writ result to report file
        result = {
                'Comparison Name': comparison['Comparison Name'],
                'Playback Flags' : comparison['Playback Flags'],
                'Reconstruction Flags' : comparison['Reconstruction Flags'],
                'Save Path' : reconstructionSavePath,
                'Comparison Path' : comparison['Save Path'],
                'Final L1/N' : 'N/A',
                'Layer 0 Iterations' : 'N/A',
                'Layer 0 Time' : str(reconMetrics['Layer 0 Time']),
                'Layer 0 L1/N' : str(reconMetrics['Layer 0 L1/N']),
                'Layer 1 Iterations' : 'N/A',
                'Layer 1 Time' : str(reconMetrics['Layer 1 Time']),
                'Layer 1 L1/N' : str(reconMetrics['Layer 1 L1/N']),
                'Comparison L1/N' : str(comparisonL1ON),
                'Comparison MSE' : str(comparisonMSE),
                'Comparison Correlation' : str(comparisonCorrelation),
                'Comparison SSIM' : str(comparisonSSIM),
                'Pass/Fail' : comparisonResult,
                'Fail Criteria' : metricsCriteria + comparisonCriteria
                }
        recordResult(reportFile, result)

        # Report pass/fail
        print '[' + comparisonResult + ']', comparison['Comparison Name']

    acquisitionClient.close()
