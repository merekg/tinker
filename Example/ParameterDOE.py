#!/usr/bin/env python3

import sys
import subprocess
import os
import numpy as np
import time
import h5py
import paramiko
import nrrd
from pyDOE2 import *
import nlopt
import copy

# Acquisition machine login
ACQUISITION_HOST = '169.254.0.1'
ACQUISITION_USER = 'company'
ACQUISITION_PASSWORD = 'pwd'

REPORT_FILE_NAME = 'comparison.csv'

TIME_MINIMUM_RECEIVE_IMAGES = 2
MAXIMUM_NLOPT_ITER = 20
DIVERGENCE_THRESHOLD = 6.0

# Constants for playback and reconstruction applications
HOME = os.path.expanduser('~')

BUILT_APPLICATIONS_PATH = HOME + '/rt3d-build/bin/'
INSTALLED_APPLICATIONS_PATH = ''
PLAYBACK_APP_PATH = INSTALLED_APPLICATIONS_PATH + 'AcquisitionPlayback'
#NOTE --sendScaled is only needed for CT inputs
PLAYBACK_SINGLE_FLAGS = ['--verbose=1', '--quitAfterPlaylist', '--writeToDatabase=false','--sendScaled','-r', '169.254.0.2', '-i', ACQUISITION_HOST, '--autoPlayback', '20', '--autoDelay', '20']
PLAYBACK_APP_COMMAND = [PLAYBACK_APP_PATH] + PLAYBACK_SINGLE_FLAGS
RECONSTRUCTION_APP_PATH = INSTALLED_APPLICATIONS_PATH + 'Reconstruction'
RECONSTRUCTION_CONFIG_PATH = '/opt/rt3d/etc/reconstruction.conf'
RECONSTRUCTION_BASE_FLAGS = ['--quitAfter1', '--configuration=' + RECONSTRUCTION_CONFIG_PATH, '--FileIO.SaveAcquisition=false']
EXPERIEMENT_HDF5_WRITE_PATH = '/media/ramdisk/optimizerOutput.h5'
RECONSTRUCTION_OPTIMIZER_FLAGS = ['--FileIO.SaveReconstruction=true', '--overrideFilePath=' + EXPERIEMENT_HDF5_WRITE_PATH]
RECONSTRUCTION_METRIC_PATH = INSTALLED_APPLICATIONS_PATH + 'Optimizer'
RECONSTRUCITON_METRIC_FLAGS = ['--generateVolumeDifferenceMetric']

# Keys for reading and writing CSV file
CSV_FILE_KEYS = [
        'Comparison Name',
        'Playback Flags',
        'Reconstruction Flags',
        'Experiment',
        'Experiment Factorial',
        'Save Path',
        'Final L1/N',
        'Numerial Analysis',
        'Visual Analysis Component',
        'MTF Coordinates',
        'CNR Coordinates']

#Keys for reading configuration file
RECONSTRUCTION_SUPPORTED_LAYERS = [
    'FL0',
    'FL1']

#List of config keys to read in default values
#Update for expanded variable set
RECONSTRUCTION_CONFIG_KEYS = [
        'ExitValue',
        'VolumeSize',
        'VolumeExtent',
        'BinFactor',
        'Lambda',
        'LearningRate']

# HDF5 paths for reconstruction volumes
HDF5_VOLUME_PATH = 'ITKImage/0/VoxelData'
HDF5_DIM_PATH = 'ITKImage/0/Dimension'
HDF5_SPACING_PATH = 'ITKImage/0/Spacing'
HDF5_SCALE_ATTRIBUTE = 'scale'

#TODO determine if non globals can be used with nlopt
acquisition = None
divergingLambda = None
groundTruthCTPath = None
fixedParameters = None
itr = 0
labels = None
experimentLayer = None
       
def runExperiment(acquisitionPath, argumentList):

    finalL1OverN = DIVERGENCE_THRESHOLD

    # Launch Reconstruction
    reconstructionCommand = []
    reconstructionCommand.append(RECONSTRUCTION_APP_PATH)
    for flag in RECONSTRUCTION_BASE_FLAGS:
        reconstructionCommand.append(flag)
    for flag in argumentList:
        reconstructionCommand.append(flag)
    #print("Recon Command: ",reconstructionCommand)
    reconProcess = subprocess.Popen(reconstructionCommand, stdout = subprocess.PIPE, universal_newlines=True)

    # Delay before launching playback
    time.sleep(1)

    client = connectAcquisitionClient()
    # Lanch Playback and wait for reconstruction to get setup
    playbackCommand = ''
    for command in PLAYBACK_APP_COMMAND:
        playbackCommand = playbackCommand + command + ' ' 
    playbackCommand = playbackCommand + ' -m ' + acquisitionPath
    #print("Playback Command: ",playbackCommand)
    playbackIn, playbackOut, playbackErr = client.exec_command(playbackCommand)
        
    reconProcessOutData, reconProcessErrData = reconProcess.communicate()
    reconProcessOutData = reconProcessOutData.split('\n')
    
    client.close()

    for line in reconProcessOutData:
        if('Recon time for layer' in line):
            finalL1OverN = float(line.split(':')[2])
    print("L1OverN Response: ", finalL1OverN)    
    return finalL1OverN

def runMetricAnalsysis(filePathGroundTruth, inputReconstruction):
    volumeDifferenceResponse = 0
    
    metricsCommand = [RECONSTRUCTION_METRIC_PATH] + RECONSTRUCITON_METRIC_FLAGS + ['--groundTruthCT='+filePathGroundTruth] + ['--inputReconstruction='+inputReconstruction]
    #print("Metrics Command: ", metricsCommand)
    metricProcess = subprocess.Popen(metricsCommand, stdout = subprocess.PIPE, universal_newlines=True)
    
    metricsProcessOutData, metricsProcessErrData = metricProcess.communicate()
    metricsProcessOutData = metricsProcessOutData.split('\n')
    for line in metricsProcessOutData:
        if('Volume Difference Metric' in line):
            volumeDifferenceResponse = float(line.split(':')[1])
            
    #TODO SSIM?
            
    print('Volume Difference Response: ', volumeDifferenceResponse)
    return volumeDifferenceResponse

def costFunction(x, grad):
    # Global variables modified in this function
    global fixedParameters, itr, divergingLambda
  
    argumentList = []
    for count in range(len(labels)):
        param  =  '--'+labels[count]+'='+str(x[count])
        argumentList.append(param)
        
    for arg in fixedParameters:
        argumentList.append(arg)
        
    for arg in RECONSTRUCTION_OPTIMIZER_FLAGS:
        argumentList.append(arg)

    # Run parameters for plug acquisition
    finalL1OverN = runExperiment(acquisition, argumentList)

    # HACK: make L1OverN inf if nan
    if np.isnan(finalL1OverN):
        finalL1OverN = sys.float_info.max
    elif finalL1OverN >= DIVERGENCE_THRESHOLD:
        divergingLambda = x[labels.index('FL0.Lambda')]
        raise nlopt.ForcedStop
    
    metricsResponse = runMetricAnalsysis(groundTruthCTPath,EXPERIEMENT_HDF5_WRITE_PATH)
            
    #Remove the output file so as to not create hdf5 write conflicts
    try:
        os.remove(EXPERIEMENT_HDF5_WRITE_PATH)
    except OSError:
        pass
    
    cost = finalL1OverN + metricsResponse

    # Print output to table
    print('%3d %2.3f %s' % (itr, cost, ' '.join(argumentList)))
    itr += 1

    return cost


def optimizeParameters(values, lowerBounds, upperBounds, initialStep):
    global itr
    
    itr = 0
    
    opt = nlopt.opt(nlopt.LN_BOBYQA, len(values))
    opt.set_lower_bounds(lowerBounds)
    opt.set_upper_bounds(upperBounds)
    opt.set_min_objective(costFunction)
    opt.set_xtol_rel(1e-3)
    opt.set_ftol_rel(1e-3)
    opt.set_initial_step(initialStep)
    opt.set_maxeval(MAXIMUM_NLOPT_ITER)

    x = opt.optimize(values)
    minf = opt.last_optimum_value()
    returnValue = opt.last_optimize_result()

    return returnValue, x
            
#Currently supporting FL0/FL1 for VolumeSize, VolumeExtent, BinFactor
#To incorporate more parameters expand the argument list for each experiment within this function
def selectParameters(configuration, timeLimit, arguments):
    
        global experimentLayer
    
        #Divide FL0/FL1 by setting one to time constraint and using configuration parameters for other
        timePairs = []
        if 'FL0' in arguments and 'FL1' in arguments:
            print("Analysis of FL0 and FL1 at the same time not implemented")
        elif 'FL0' in arguments:
            timePair = []
            timePair.append(timeLimit)
            timePair.append(0)
            timePairs.append(timePair)
        elif 'FL1' in arguments:
            timePair = []
            timePair.append(configuration['FL0']['ExitValue'])
            timePair.append(timeLimit)
            timePairs.append(timePair)
        
        factors = []
        
        #Generate the fractional fractorial list
        if 'FL0' in arguments:
            if 'VolumeSize' in arguments:
                factors.append('a')
                factors.append('b')
                
            if 'VolumeExtent' in arguments:
                factors.append('c')
                factors.append('d')
                
            if 'BinFactor' in arguments:
                factors.append('e')
            
        factorString = ''
        for factor in factors:
            factorString = factorString + factor + ' '
        fractional = fracfact(factorString)
        
        experiments = []
        
        #Generate all arguments for experiments
        for experiment in fractional:
            for timePair in timePairs:
                experimentFactorialPair = []
                experimentList = []
                
                arg = '--FL0.ExitCriteria=MS'
                experimentList.append(arg)
                arg = '--FL0.ExitValue=' + str(timePair[0])
                experimentList.append(arg)
                arg = '--FL1.ExitCriteria=MS'
                experimentList.append(arg)
                arg = '--FL1.ExitValue=' + str(timePair[1])
                experimentList.append(arg)
                
                volumeSize = copy.deepcopy(configuration['FL0']['VolumeSize'])
                volumeExtent = copy.deepcopy(configuration['FL0']['VolumeExtent'])
                binFactor = int(copy.deepcopy(configuration['FL0']['BinFactor'][0]))
            
                fractionalExpansionIndex = 0
                if 'VolumeSize' in arguments:
                    deltaXY = round(np.sqrt(configuration['FL0']['VolumeSize'][0]))
                    deltaXY = deltaXY + (deltaXY % 2)
                    
                    deltaZ = round(np.sqrt(configuration['FL0']['VolumeSize'][1]))
                    deltaZ = deltaZ + (deltaZ % 2)
                    #Start with an estimate lower than the current to include current as +1
                    volumeSize[0] -= deltaXY
                    volumeSize[1] -= deltaXY
                    volumeSize[2] -= deltaZ
                
                    volumeSize[0] = int(volumeSize[0]) + int(deltaXY * experiment[fractionalExpansionIndex])
                    volumeSize[1] = int(volumeSize[0])
                    fractionalExpansionIndex += 1
                    
                    volumeSize[2] = int(volumeSize[2]) + int(deltaZ * experiment[fractionalExpansionIndex])
                    fractionalExpansionIndex += 1
                    
                param = str(volumeSize[0]) + ',' + str(volumeSize[1]) + ',' + str(volumeSize[2])
                arg = experimentLayer + '.VolumeSize=' + param
                experimentList.append(arg)
                    
                if 'VolumeExtent' in arguments:
                    deltaXY = round(np.sqrt(configuration['FL0']['VolumeExtent'][0]))
                    deltaXY = deltaXY + (deltaXY % 2)
                    
                    deltaZ = round(np.sqrt(configuration['FL0']['VolumeExtent'][2]))
                    deltaZ = deltaZ + (deltaZ % 2)
                    #Start with an estimate lower than the current to include current as +1
                    volumeExtent[0] -= deltaXY
                    volumeExtent[1] -= deltaXY
                    volumeExtent[2] -= deltaZ
                    
                    
                    volumeExtent[0] = volumeExtent[0] + int(deltaXY * experiment[fractionalExpansionIndex])
                    volumeExtent[1] = volumeExtent[0]
                    fractionalExpansionIndex += 1
                    
                    volumeExtent[2] = volumeExtent[2] + int(deltaZ * experiment[fractionalExpansionIndex])
                    fractionalExpansionIndex += 1
                    
                param = str(volumeExtent[0]) + ',' + str(volumeExtent[1]) + ',' + str(volumeExtent[2])
                arg = experimentLayer + '.VolumeExtent=' + param
                experimentList.append(arg)
                
                param = str(0) + ',' + str(0) + ',' + str(volumeExtent[2]/2)
                arg = experimentLayer + '.VolumePosition=' + param
                experimentList.append(arg)
                    
                if 'BinFactor' in arguments:
                    binFactor = int(binFactor + experiment[fractionalExpansionIndex])
                    fractionalExpansionIndex += 1
                    
                arg = experimentLayer + '.BinFactor=' + str(binFactor)
                experimentList.append(arg)
                
                if 'FL1' in arguments:
                    arg = '--FL0.MultiResSteps=2'
                    experimentList.append(arg)
                else:
                    arg = '--FL0.MultiResSteps=1'
                    experimentList.append(arg)
                    
                        
                experimentFactorialPair.append(experimentList)
                experimentFactorialPair.append(experiment)
                experiments.append(experimentFactorialPair)
                
        return experiments

def printHelp():

    print('Summary: This application will select parameter sets based on given criteria, the current options include: (FL0, FL1, VolumeSize, VolumeExtent, BinFactor)')
    print('Usage: python IQComparisonTool.py <output-folder> <config-file-path> <acquisition-file> <ground-truth-CT> <time-constraint-ms> <layer-list> <parameters-to-select>')
    

def connectAcquisitionClient():

    try:
        client = paramiko.client.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(ACQUISITION_HOST, username=ACQUISITION_USER, password=ACQUISITION_PASSWORD)
    except Exception as e:
        print('Failed to connect to Acquisition PC', str(e))
        printHelp()
        exit()

    return client

def loadHDF5Volume(volumePath):
    volume = np.array([])
    dimensions = np.array([])
    spacings = np.array([])

    try:
        with h5py.File(volumePath, 'r') as volumeFile:
            voxelData = volumeFile[HDF5_VOLUME_PATH][:].astype(float)
            dimensions = volumeFile[HDF5_DIM_PATH][:].astype(float)
            spacings = volumeFile[HDF5_SPACING_PATH][:].astype(float)
            scaleFactor = float(volumeFile[HDF5_VOLUME_PATH].attrs[HDF5_SCALE_ATTRIBUTE])
            voxelData.reshape([int(dimensions[0]),int(dimensions[1]),int(dimensions[2])])
            volume = voxelData / scaleFactor
    except Exception as e:
        print(('Failed to load volume: ' + volumePath + ' : ' + str(e) + '\n'))

    return volume,dimensions,spacings

def loadNRRDVolume(volumePath):
    try:
        volume, header = nrrd.read(volumePath)
        dimensions = header['sizes']
        spacings = header['spacings']
    except Exception as e:
        print(('Failed to load nrrd volume: ' + volumePath + ' : ' + str(e) + '\n'))
    
    return volume,(dimensions*spacings)
    
    
def createSaveFolder(outputFolder):

    try:
        saveFolder = os.path.join(outputFolder, "Parameter_DOE")
        os.mkdir(saveFolder)
    except Exception as e:
        print('Failed to create output folder', str(e))
        printHelp()
        exit()

    return saveFolder

def readConfigurationFile(configFilePath):
    f = open(configFilePath, "r")
  
    currentLayerRead = ''
    readLayer = False
    defaultParameters = {}
    for layer in RECONSTRUCTION_SUPPORTED_LAYERS:
        defaultParameters[layer] = {}
        
    for line in f:
        if '[' in line:
            readLayer = False
            for layer in RECONSTRUCTION_SUPPORTED_LAYERS:
                if layer in line:
                    currentLayerRead = layer
                    readLayer = True
                    
        if not readLayer:  
            continue
        
        for key in RECONSTRUCTION_CONFIG_KEYS:
            if key in line:
                paramterSegments = line.split('=')[1].split(',')
                
                parameter = []
                for segment in paramterSegments:
                    parameter.append(float(segment))
                defaultParameters[currentLayerRead][key] = parameter
                
                
    return defaultParameters
                
    

def openReportFile(saveFolder, name):

    try:
        reportFilePath = os.path.join(saveFolder, name)
        print(reportFilePath)
        reportFile = open(reportFilePath, 'w')
        reportFile.write(';'.join(CSV_FILE_KEYS))
        reportFile.write('\n')
    except Exception as e:
        print('Failed to open report file', str(e))
        exit()

    return reportFile

def recordResult(reportFile, result):
    for key in result.keys():
        reportFile.write(result[key] + ';')
    reportFile.write('\n')

def main():

    global acquisition, groundTruthCTPath, labels, experimentLayer, fixedParameters, divergingLambda
    
    print("Start time: " + str(time.time()))
    
    # Get commandline arguments
    if "help" in sys.argv:
        printHelp()
        exit()
        
    outputFolder = sys.argv[1]
    configFilePath = sys.argv[2]
    acquisition = sys.argv[3]
    groundTruthCTPath = sys.argv[4]
    timeLimit = int(sys.argv[5])
    arguments = sys.argv[6:]
        
    #Generate the baseline
    baselineReconArgs = []
    if 'FL1' in arguments:
        arg = '--FL0.MultiResSteps=2'
        experimentLayer = '--FL1'
        baselineReconArgs.append(arg)
    else:
        arg = '--FL0.MultiResSteps=1'
        experimentLayer = '--FL0'
        baselineReconArgs.append(arg)
    
    configuration = readConfigurationFile(configFilePath)
    
    experiments = selectParameters(configuration, timeLimit, arguments)
    saveFolder = createSaveFolder(outputFolder)
    outputFile = openReportFile(saveFolder, REPORT_FILE_NAME)
    experimentCount = 0
    
    labels = []
    values = []
    lowerBounds = []
    upperBounds = []
    initialStep = []

    # Get optimization sets
    if 'FL0' in arguments:
        labels += ['FL0.Lambda', 'FL0.LearningRate']
        values = [configuration['FL0']['Lambda'][0], configuration['FL0']['LearningRate'][0]]
        lowerBounds = [0.0001, 0.2]
        upperBounds = [0.3, 0.99]
        initialStep = [0.05, 0.05]
    elif 'FL1' in arguments:
        labels += ['FL1.Lambda', 'FL1.LearningRate']
        values = [configuration['FL1']['Lambda'][0], configuration['FL1']['LearningRate'][0]]
        lowerBounds = [0.0001, 0.2]
        upperBounds = [0.3, 0.99]
        initialStep = [0.05, 0.05]
        
    arg = experimentLayer + '.ExitCriteria=MS'
    baselineReconArgs.append(arg)
    arg = experimentLayer + '.ExitValue=' + str(timeLimit)
    baselineReconArgs.append(arg)
        
    reconName = 'baseline.h5'
    reconstructionSavePath = os.path.abspath(os.path.join(saveFolder, reconName))
    baselineReconArgs.append('--overrideFilePath=' + reconstructionSavePath)
    
    baselineExperimentFactorial = ''
    for i in range(len(experiments[0][1])):
        baselineExperimentFactorial = baselineExperimentFactorial + ' 1.0'
            
    baselineL1OverN = runExperiment(acquisition, baselineReconArgs)
    baselineNumericalMetric = runMetricAnalsysis(groundTruthCTPath,reconstructionSavePath)
        
    reconstructionFlags = ''
    for flag in baselineReconArgs:
        reconstructionFlags = reconstructionFlags + ' ' + flag
    result = {
            'Comparison Name': reconName,
            'Playback Flags' : "-m " + acquisition,
            'Reconstruction Flags' : reconstructionFlags,
            'Experiment' : ' ',
            'Experiment Factorial': baselineExperimentFactorial,
            'Save Path' : reconstructionSavePath,
            'Final L1/N': str(baselineL1OverN),
            'Numerial Analysis' : str(baselineNumericalMetric),
            'Visual Analysis Component' : '',
            'MTF Coordinates' : '',
            'CNR Coordinates' : ''
            }
    recordResult(outputFile, result)
    print("baselineR: ", result)
        
    #Optimize lambda and learning rate for each experiment
    for experiment in experiments:
        fixedParameters = experiment[0]
        
        returnVal = -1
        estimateValues = copy.deepcopy(values) 
        estimateInitialStep = copy.deepcopy(initialStep)
        upperBoundItr = copy.deepcopy(upperBounds)
        print("Estimate: ", estimateValues, ' ', estimateInitialStep)
        
        #If the optimizer fails too many times it is unlikely to find any solution (at that point lambda start is 0.001875)
        executionAttempts = 0
        while returnVal < 0 and executionAttempts < 6:
            executionAttempts += 1
            #determine an estimate for the optimal lambda and learning rate
            try:
                returnVal, optimizedParams = optimizeParameters(estimateValues, lowerBounds, upperBoundItr, estimateInitialStep)
            except nlopt.ForcedStop as e:
                upperBoundItr[0] = divergingLambda * 2
                #If divergence was encountered restart with a lower estimate
                estimateValues[0] = estimateValues[0]/2.0
                estimateInitialStep = estimateValues[0]
        
        experiment[0].append(experimentLayer + '.Lambda=' + str(optimizedParams[0]))
        experiment[0].append(experimentLayer + '.LearningRate=' + str(optimizedParams[1]))
        
        reconName = "Experiment" + str(experimentCount) + ".h5"
        reconstructionSavePath = os.path.abspath(os.path.join(saveFolder, reconName))
        experiment[0].append('--overrideFilePath=' + reconstructionSavePath)
        
        #Run the final Reconstruction to save into the result folder
        finalL1OverN = runExperiment(acquisition, experiment[0])
        finalNumericalMetric = runMetricAnalsysis(groundTruthCTPath,reconstructionSavePath)
    
        reconstructionFlags = ''
        for flag in experiment[0]:
            reconstructionFlags = reconstructionFlags + ' ' + flag
        
        experimentFactorial = ''
        for factor in experiment[1]:
            experimentFactorial = experimentFactorial + ' ' + str(factor)
            
        experimentList = ''
            
        # Record and writ result to report file
        result = {
                'Comparison Name': reconName,
                'Playback Flags' : "-m " + acquisition,
                'Reconstruction Flags' : reconstructionFlags,
                'Experiment' : ' '.join(arguments),
                'Experiment Factorial': experimentFactorial,
                'Save Path' : reconstructionSavePath,
                'Final L1/N': str(finalL1OverN),
                'Numerial Analysis' : str(finalNumericalMetric),
                'Visual Analysis Component' : '',
                'MTF Coordinates' : '',
                'CNR Coordinates' : ''
                }
        recordResult(outputFile, result)
            
        experimentCount = experimentCount + 1

    print("Completion time: " + str(time.time()))


if __name__ == "__main__" :
    main()
