import subprocess
import os
import sys
import shutil
import nlopt
import h5py
import numpy as np
import time
import itertools

# Import IQ Metrics library TODO: Find more elegant way of doing this
sys.path.append('../ImageQualityTool')
import IQ_Metrics as iqm

# General constanst for application
HOME = os.path.expanduser('~')
OPTIMIZER_DATA_FOLDER = HOME + '/Optimizer'

# Constants for acquisition files
ACQUISITION_SINGLE_PATH = OPTIMIZER_DATA_FOLDER + '/single.h5'
ACQUISITION_DUAL_PATH = OPTIMIZER_DATA_FOLDER + '/NormalDose_Rando_Dual.i+.0.h5'

# Constants for existing volume
EXISTING_VOLUME_PATH = OPTIMIZER_DATA_FOLDER + '/NormalDose_Rando_Dual_First_Half.h5'
EXISTING_ACQUISITION_PATH = OPTIMIZER_DATA_FOLDER + '/NormalDose_Rando_Dual.i-.0.h5'

# Constants for computation of metrics
MTF_SINGLE_COORDINATES = [430, 459, 54]
MTF_DUAL_COORDINATES = [439, 452, 49]
#CNR_SINGLE_COORDINATES = [430, 459, 54, 384, 383, 54]
#CNR_DUAL_COORDINATES = [439, 452, 49, 393, 377, 49]
CNR_SINGLE_COORDINATES = [215, 291, 146, 355, 654, 125] # For Rando
CNR_DUAL_COORDINATES = [215, 291, 146, 355, 654, 125] # For Rando

# Constants for files management
RECONSTRUCTION_SHM_PATH = '/dev/shm/nVReconstruction'
configFileLocation = HOME+'/rt3d/Apps/Reconstruction/reconstruction.conf'

# Constants for playback and reconstruction applications
BUILT_APPLICATIONS_PATH = HOME + '/rt3d-build/bin/'
INSTALLED_APPLICATIONS_PATH = ''
APPLICATIONS_PATH = INSTALLED_APPLICATIONS_PATH
PLAYBACK_APP_PATH = APPLICATIONS_PATH + 'AcquisitionPlayback'
PLAYBACK_BASE_FLAGS = ['--verbose=1', '--quitAfterPlaylist']
PLAYBACK_APP_COMMAND = [PLAYBACK_APP_PATH] + PLAYBACK_BASE_FLAGS
RECONSTRUCTION_APP_PATH = APPLICATIONS_PATH + 'Reconstruction'
RECONSTRUCTION_BASE_FLAGS = ['--configuration='+configFileLocation, '--quitAfter1']
RECONSTRUCTION_APP_COMMAND = [RECONSTRUCTION_APP_PATH] + RECONSTRUCTION_BASE_FLAGS

# Constants for file output
OPTIMIZER_OUTPUT_FOLDER = OPTIMIZER_DATA_FOLDER + '/Output'
OPTIMIZER_LOG_NAME = 'optimizer.log'
OPTIMIZER_LOG_FOLDER = OPTIMIZER_OUTPUT_FOLDER
OPTIMIZER_LOG_PATH = os.path.join(OPTIMIZER_LOG_FOLDER, OPTIMIZER_LOG_NAME)
BEST_RECONSTRUCTION_PATH = OPTIMIZER_OUTPUT_FOLDER + '/BestRecon.h5'
BEST_RECONSTRUCTION_LOG_PATH = OPTIMIZER_OUTPUT_FOLDER + '/BestRecon.out'
RECONSTRUCTION_FOLDER = '/opt/rt3d/save'
RECONSTRUCTION_PATH = RECONSTRUCTION_FOLDER + '/recon0.h5'
RECONSTRUCTION_LOG_PATH = '/media/ramdisk/reconstruction.out'

EXIT_VALUE_MAP = {
        nlopt.SUCCESS: "generic success",
        nlopt.STOPVAL_REACHED: "stopval reached",
        nlopt.FTOL_REACHED: "function absolute or relative tolerance reached",
        nlopt.XTOL_REACHED: "X absolute or relative tolerance reached",
        nlopt.MAXEVAL_REACHED: "maxevl reached",
        nlopt.MAXTIME_REACHED: "maxtime reached"
}

# Global variables (and some constants) TODO: clean up
finalL1OverN = None
itr = 1
argumentString = []
layer = None
bestX = []
minCost = sys.float_info.max
bestParameters = []
cnrCoordinates = None
mtfCoordinates = None
plugAcquisition = None
labels = None
TOTAL_TIME = 45000

def clearFolder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def runExperiment(acquisitionPath, argumentString):

    global finalL1OverN

    # Launch Reconstruction
    reconstructionCommand = RECONSTRUCTION_APP_COMMAND + argumentString
    reconProcess = subprocess.Popen(reconstructionCommand, stdout = subprocess.PIPE, stdin = subprocess.PIPE, stderr = subprocess.PIPE)

    # Delay before launching playback
    time.sleep(1)

    # Lanch Playback and wait for reconstruction to get setup
    playbackCommand = PLAYBACK_APP_COMMAND + ['-m', acquisitionPath]
    playbackProcess = subprocess.Popen(playbackCommand, stdout = subprocess.PIPE, stdin = subprocess.PIPE, stderr = subprocess.PIPE)
    time.sleep(8) #TODO: This can be reduced for singles

    # Send image over playback
    playbackProcess.communicate(input = ('0\n').encode())

    # Get data back
    reconProcessOutData, reconProcessErrData = reconProcess.communicate()
    reconProcessOutData = reconProcessOutData.split('\n')

    for line in reconProcessOutData:
        if('Recon time for layer' in line):
            finalL1OverN = float(line.split(':')[2])

def costFunction(x, grad):

    # Global variables modified in this function
    global bestX, minCost, argumentString, bestParameters, itr, finalL1OverN

    # Get argument lists
    if(len(x) < len(labels)): # If there are less variables than labels it is the Timing Split test
        l0Time = TOTAL_TIME*x[4]
        l1Time = TOTAL_TIME*(1-x[4])
        variableParameters = ['--'+labels[0]+'='+str(x[0]),'--'+labels[1]+'='+str(x[1]),'--'+labels[2]+'='+str(x[2]),'--'+labels[3]+'='+str(x[3]),'--'+labels[4]+'='+str(int(l0Time)),'--'+labels[5]+'='+str(int(l1Time))]
    else:   
        variableParameters = ['--'+labels[count]+'='+str(x[count]) for count in xrange(len(labels))]
    argumentString = variableParameters + fixedParameters

    # Clear output at completion of computations
    clearFolder(RECONSTRUCTION_FOLDER)

    # Run parameters for plug acquisition
    runExperiment(plugAcquisition, argumentString)

    # Load output file and get metrics
    dataset = h5py.File(RECONSTRUCTION_PATH, 'r')
    #mtf = iqm.runMTF2D(dataset, mtfCoordinates)
    #cnr = iqm.runCNR(dataset, cnrCoordinates)
    mtf = 0
    cnr = 0

    # HACK: make MTF 0 if big (it shouldn't get that big)
    if mtf > 0.4:
        mtf = 0.0

    # HACK: make CNR 0 if nan
    if np.isnan(cnr):
        cnr = 0.0

    # HACK: make L1OverN inf if nan
    if np.isnan(finalL1OverN):
        finalL1OverN = sys.float_info.max

    # Compute cost function
    #cost = -5*mtf - 0.01*cnr + 50*finalL1OverN
    #cost = -5*mtf - 0.04*cnr + 50*finalL1OverN
    #cost = -5*mtf - 0.04*cnr + 10*finalL1OverN
    cost = finalL1OverN
    #cost = -5*mtf - 0.04*cnr + 50*finalL1OverN

    # Track minimum cost
    if minCost > cost:
        minCost = cost
        bestParameters = [cost, mtf, cnr, finalL1OverN, ' '.join(argumentString)]
        clearFolder(OPTIMIZER_OUTPUT_FOLDER)
        shutil.move(RECONSTRUCTION_PATH, BEST_RECONSTRUCTION_PATH)
        shutil.move(RECONSTRUCTION_LOG_PATH, BEST_RECONSTRUCTION_LOG_PATH)

    # Clear output at completion of computations
    clearFolder(RECONSTRUCTION_FOLDER)

    # Print output to table
    print('%3d %2.3f %s' % (itr, cost, ' '.join(variableParameters)))
    itr += 1

    return cost


def optimizeParameters():

    opt = nlopt.opt(nlopt.LN_COBYLA, len(values))
    opt.set_lower_bounds(lowerBounds)
    opt.set_upper_bounds(upperBounds)
    opt.set_min_objective(costFunction)
    opt.set_xtol_rel(1e-3)
    opt.set_ftol_rel(1e-3)
    opt.set_initial_step(initialStep)

    x = opt.optimize(values)
    minf = opt.last_optimum_value()
    returnValue = opt.last_optimize_result()

    return returnValue


def getExperiments(arguments):

    # List parameters to be tested
    parameters = []

    if 'FL0' in arguments:
        #parameters.append(('FL0.VolumeSize', ['384,384,192']))
        #parameters.append(('FL0.BinFactor', ['2']))
        parameters.append(('FL0.TvIterations', ['0']))
        parameters.append(('FL0.ExitCriteria', ['ArtViews']))
        parameters.append(('FL0.ExitValue', [26000]))
        parameters.append(('FL0.MultiResSteps', ['1']))

    if 'ISL0' in arguments:
        #parameters.append(('ISL0.VolumeSize', ['384,384,192']))
        #parameters.append(('ISL0.BinFactor', ['2']))
        parameters.append(('ISL0.TvIterations', ['0']))
        parameters.append(('ISL0.ExitCriteria', ['ArtViews']))
        parameters.append(('ISL0.ExitValue', [26000]))
        parameters.append(('ISL0.MultiResSteps', ['1']))

    if 'ISL1' in arguments:
        #parameters.append(('ISL1.VolumeSize', ['384,384,96']))
        #parameters.append(('ISL1.BinFactor', ['1']))
        parameters.append(('ISL1.TvIterations', ['0']))
        parameters.append(('ISL1.ExitCriteria', ['ArtViews']))
        parameters.append(('ISL1.ExitValue', [25000]))
        parameters.append(('ISL0.MultiResSteps', ['2']))

    if 'ISLTime' in arguments:
        parameters.append(('ISL0.ExitCriteria', ['MS']))
        parameters.append(('ISL1.ExitCriteria', ['MS']))
        parameters.append(('ISL0.MultiResSteps', ['2']))

    if 'IDL0' in arguments:
        #parameters.append(('IDL0.VolumeSize', ['384,384,192']))
        #parameters.append(('IDL0.BinFactor', ['2']))
        parameters.append(('IDL0.TvIterations', ['0']))
        parameters.append(('IDL0.ExitCriteria', ['MS']))
        parameters.append(('IDL0.ExitValue', [15000]))
        parameters.append(('IDL0.MultiResSteps', ['1']))

    if 'IDL1' in arguments:
        parameters.append(('IDL1.VolumeSize', ['384,384,96']))
        parameters.append(('IDL1.BinFactor', ['1']))
        parameters.append(('IDL1.TvIterations', ['0']))
        parameters.append(('IDL1.ExitCriteria', ['MS']))
        parameters.append(('IDL1.ExitValue', [15000]))
        parameters.append(('IDL0.MultiResSteps', ['2']))

    if 'IDL2' in arguments:
        parameters.append(('IDL2.VolumeSize', ['768,768,256']))
        parameters.append(('IDL2.BinFactor', ['0']))
        parameters.append(('IDL2.TvIterations', ['0']))
        parameters.append(('IDL2.ExitCriteria', ['ArtViews']))
        parameters.append(('IDL2.ExitValue', [10000]))
        parameters.append(('IDL0.MultiResSteps', ['3']))

    # General settings for all cases
    parameters.append(('FileIO.SaveReconstruction', ['1']))
    #parameters.append(('Corrections.ApplyOffsetCorrection', ['false']))
    #parameters.append(('Corrections.ApplyTechniqueCorrection', ['false']))
    #parameters.append(('Corrections.ApplySignalInAirCorrection', ['false']))
    #parameters.append(('Corrections.ApplyLogging', ['false']))
    #parameters.append(('Corrections.ApplyDeadPixelCorrection', ['false']))
    #parameters.append(('Filters.MedianFilterDiameter', ['0']))

    # Create lists for parameter command line strings
    parameterSets = []

    # Iterate over parameters and format strings for command line
    for label, values in parameters:
        parameterStrings = []
        for value in values:
            parameterStrings.append('--' + label + '=' + str(value))
        parameterSets.append(parameterStrings)

    # Produce all combinations of the parameters
    return itertools.product(*parameterSets)


def main():

    print "Start time: " + str(time.time())

    # Global variables for main TODO: get rid of global variables
    global labels, values, lowerBounds, upperBounds, initialStep, fixedParameters, cnrCoordinates, mtfCoordinates, RECONSTRUCTION_APP_COMMAND, plugAcquisition, layer

    # Clear output folders and shared memory
    clearFolder(RECONSTRUCTION_FOLDER)
    try:
        os.remove(RECONSTRUCTION_SHM_PATH)
    except OSError:
        pass

    # Get commandline arguments
    arguments = sys.argv[1:]

    # Get combinations of experiments to run
    experiments = getExperiments(arguments)

    labels = []
    values = []
    lowerBounds = []
    upperBounds = []
    initialStep = []

    # Get optimization sets
    if 'FL0' in arguments:
        labels += ['FL0.Lambda', 'FL0.LearningRate']
        values = [0.1283, 0.9553]
        lowerBounds = [0.0001, 0.2]
        upperBounds = [0.5, 0.99]
        initialStep = [0.05, 0.05]
        cnrCoordinates = CNR_SINGLE_COORDINATES 
        mtfCoordinates = MTF_SINGLE_COORDINATES
        plugAcquisition = ACQUISITION_SINGLE_PATH  
        layer = '0'

    if 'ISL0' in arguments:
        labels += ['ISL0.Lambda', 'ISL0.LearningRate']
        values = [0.1283, 0.9553]
        lowerBounds = [0.0001, 0.2]
        upperBounds = [0.5, 0.99]
        initialStep = [0.05, 0.05]
        cnrCoordinates = CNR_SINGLE_COORDINATES 
        mtfCoordinates = MTF_SINGLE_COORDINATES
        plugAcquisition = ACQUISITION_SINGLE_PATH  
        layer = '0'

    if 'ISL1' in arguments:
        labels += ['ISL1.Lambda', 'ISL1.LearningRate']
        values = [0.3585, 0.8912]
        lowerBounds = [0.0001, 0.2]
        upperBounds = [0.5, 0.99]
        initialStep = [0.05, 0.05]
        cnrCoordinates = CNR_SINGLE_COORDINATES 
        mtfCoordinates = MTF_SINGLE_COORDINATES
        plugAcquisition = ACQUISITION_SINGLE_PATH  
        layer = '1'

    if 'ISLTime' in arguments:
        labels += ['ISL0.Lambda', 'ISL0.LearningRate', 'ISL1.Lambda', 'ISL1.LearningRate', 'ISL0.ExitValue', 'ISL1.ExitValue']
        values = [0.27, 0.9, 0.35, 0.9, 0.333]
        lowerBounds = [0.0001, 0.2, 0.0001, 0.2, 0.167]
        upperBounds = [0.5, 0.99, 0.5, 0.99, 0.833]
        initialStep = [0.05, 0.05, 0.05, 0.05, 0.1]
        cnrCoordinates = CNR_SINGLE_COORDINATES 
        mtfCoordinates = MTF_SINGLE_COORDINATES
        plugAcquisition = ACQUISITION_SINGLE_PATH  
        layer = '2'

    if 'IDL0' in arguments:
        labels += ['IDL0.Lambda', 'IDL0.LearningRate']
        values += [0.15, 0.9]
        lowerBounds += [0.0001, 0.2]
        upperBounds += [0.5, 0.99]
        initialStep += [0.05, 0.05]
        cnrCoordinates = CNR_DUAL_COORDINATES
        mtfCoordinates = MTF_DUAL_COORDINATES
        plugAcquisition = ACQUISITION_DUAL_PATH 
        RECONSTRUCTION_APP_COMMAND += ['--existing-volume', EXISTING_VOLUME_PATH]
        RECONSTRUCTION_APP_COMMAND += ['--i-minusProj', EXISTING_ACQUISITION_PATH]
        layer = '0'

    if 'IDL1' in arguments:
        labels += ['IDL1.Lambda', 'IDL1.LearningRate']
        values += [0.1, 0.9]
        lowerBounds += [0.0001, 0.2]
        upperBounds += [0.5, 0.99]
        initialStep += [0.005, 0.05]
        cnrCoordinates = CNR_DUAL_COORDINATES
        mtfCoordinates = MTF_DUAL_COORDINATES
        plugAcquisition = ACQUISITION_DUAL_PATH 
        RECONSTRUCTION_APP_COMMAND += ['--existing-volume', EXISTING_VOLUME_PATH]
        RECONSTRUCTION_APP_COMMAND += ['--i-minusProj', EXISTING_ACQUISITION_PATH]
        layer = '1'

    if 'IDL2' in arguments:
        labels += ['IDL2.Lambda', 'IDL2.LearningRate']
        values += [0.01, 0.8]
        lowerBounds += [0.0001, 0.2]
        upperBounds += [0.5, 0.99]
        initialStep += [0.03, 0.05]
        cnrCoordinates = CNR_DUAL_COORDINATES
        mtfCoordinates = MTF_DUAL_COORDINATES
        plugAcquisition = ACQUISITION_DUAL_PATH 
        RECONSTRUCTION_APP_COMMAND += ['--existing-volume', EXISTING_VOLUME_PATH]
        RECONSTRUCTION_APP_COMMAND += ['--i-minusProj', EXISTING_ACQUISITION_PATH]
        layer = '2'


    # Format table for printing data
    print('Itr Cost  Parameters')

    # For each set experiment run the optimizer
    for experiment in experiments:
        fixedParameters = list(experiment)
        returnValue = optimizeParameters()
        print "Experiment completed with " + EXIT_VALUE_MAP[returnValue] + " return value"

    print 'Best Parameters:'
    print bestParameters
    print "Completion time: " + str(time.time())


if __name__ == "__main__" :
    main()

