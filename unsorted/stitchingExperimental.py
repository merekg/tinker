#!/usr/bin/env python3

import subprocess as spr
import pymysql
import h5py
import numpy as np
import sys
import os

# Wheel constant
WHEEL_CIRCUMFERENCE_MM = 630
CM_TO_MM = 10

#Database connection variables
MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PASSWORD = ''
MYSQL_DATABASE = 'nView'

# Application paths
RECONSTRUCTION_APP = 'Reconstruction'
RECONSTRUCTION_STITCHING_FLAGS = ['--DisplayVolumeGeometry.VolumeSize=832,416,416', '--DisplayVolumeGeometry.VolumeExtent=600,300,300', '--quitAfterDual', '--FileIO.SaveAcquisition=false']
PLAYBACK_APP = 'AcquisitionPlayback'
PLAYBACK_STITCHING_FLAGS = ['--writeToDatabase', 'false', '--quitAfterPlaylist', '--autoPlayback', '10', '--autoDelay', '2']

# File save path
ACQ_FILE_PATH = '/opt/rt3d/save/acquisitions/acq'

def printHelp():

    print ('Summary: This application will reconstruct a stitched image from the two most recent acquisitions. The wheel position of the system must be enetered for each acquisition.')
    print ('Usage: python stitchingExperimental.py <System-movement: l, r> <wheel-position-1> <wheel-position-2>')

def processArguments(argumentsList):

    try:
        assert len(argumentsList) == 4
        translationDirection = argumentsList[1]
        if(translationDirection != '-' and translationDirection != '+'):
            print('Error: Invalid Direction entered.')
            printHelp()
            exit()
            
        wheelPositionStartCM = float(argumentsList[2])
        wheelPositionEndCM = float(argumentsList[3])
    except Exception as e:
        print ('Invalid command line arguments', str(e))
        printHelp()
        exit()

    return translationDirection, wheelPositionStartCM, wheelPositionEndCM

def getAcquisitionFiles():

    # Get acqFiles from database\
    dbConnection = pymysql.connect(host = MYSQL_HOST, user = MYSQL_USER, password = MYSQL_PASSWORD, database = MYSQL_DATABASE)

    with dbConnection.cursor() as cursor:
        cursor.execute('select max(ProcedureID) from procedures')
        procedureID = cursor.fetchone()[0]
        cursor.execute('select AcquisitionID from proceduresAcquisitionsJunction where ProcedureID = %s', str(procedureID))
        acqIDs = cursor.fetchall()
        cursor.close()

    acqFiles = [ACQ_FILE_PATH + str(acqIDs[-2][0]) + '.h5', ACQ_FILE_PATH + str(acqIDs[-1][0]) + '.h5']

    return acqFiles

def getPlaybackCommand(acqFiles):

    acqFlags = ['-m', acqFiles[0], '-p', acqFiles[1]]
    playbackFlags = acqFlags + PLAYBACK_STITCHING_FLAGS

    return ' '.join([PLAYBACK_APP] + playbackFlags)

if __name__ == '__main__':

    # Setup files and infrastructure
    translationDirectrion, wheelPositionStartCM, wheelPositionEndCM = processArguments(sys.argv)

    # Calculate translation difference
    if(translationDirectrion == '+'): # positive movement
        deltaTranslationMM = (wheelPositionEndCM - wheelPositionStartCM)*CM_TO_MM
        if (wheelPositionStartCM > wheelPositionEndCM): # Cross over 0
            deltaTranslationMM += WHEEL_CIRCUMFERENCE_MM
    else: # negative movement 
        deltaTranslationMM = (wheelPositionStartCM - wheelPositionEndCM)*CM_TO_MM
        if (wheelPositionStartCM < wheelPositionEndCM): # Cross over 0
            deltaTranslationMM += WHEEL_CIRCUMFERENCE_MM
        deltaTranslationMM = -1*deltaTranslationMM

    # Stop CArmPowerManager service and shutdown carm
    print('Turning off Carm')
    os.system('systemctl stop CArmPowerManager.service')
    spr.call('CArmPowerOff.sh')

    # Stop Recon service
    print('Stopping Reconstruction service')
    os.system('systemctl stop Reconstruction.service')

    # Get and modify acquisition
    acqFiles = getAcquisitionFiles()

    with h5py.File(acqFiles[0],'r+') as h5File:
        pose1 = h5File['ITKImage/0/MetaData/imagePnO'][:]
        modPose = pose1.copy()
        modPose[0] += deltaTranslationMM / 2.0
        h5File['ITKImage/0/MetaData/imagePnO'][:] = modPose
        h5File.close()

    with h5py.File(acqFiles[1],'r+') as h5File:
        pose2 = h5File['ITKImage/0/MetaData/imagePnO'][:]
        modPose = pose2.copy()
        modPose[0] -= deltaTranslationMM / 2.0
        h5File['ITKImage/0/MetaData/imagePnO'][:] = modPose
        type = h5File['ITKImage/0/VoxelData/'].attrs['Type']
        h5File['ITKImage/0/VoxelData/'].attrs['Type'] = 'InitialOne'
        h5File.close()

    # Get reconstruction and playback commands
    reconCommand = [RECONSTRUCTION_APP] + RECONSTRUCTION_STITCHING_FLAGS
    playbackCommand = getPlaybackCommand(acqFiles)

    # Launch Reconstruction
    print ('Running reconstruction with command: ' + ' '.join(reconCommand) + '\n')
    reconProcess = spr.Popen(reconCommand, stdout=spr.PIPE, stdin=spr.PIPE, stderr=spr.PIPE)

    # Launch Playback
    print ('Running playback with command: ' + playbackCommand + '\n')
    os.system(playbackCommand)

    # Block until Reconstruction completes
    reconOut, reconErr = reconProcess.communicate()

    # Unmodify acquisitions
    with h5py.File(acqFiles[0],'r+') as h5File:
        h5File['ITKImage/0/MetaData/imagePnO'][:] = pose1
        h5File.close()

    with h5py.File(acqFiles[1],'r+') as h5File:
        h5File['ITKImage/0/MetaData/imagePnO'][:] = pose2
        h5File['ITKImage/0/VoxelData/'].attrs['Type'] = type
        h5File.close()
