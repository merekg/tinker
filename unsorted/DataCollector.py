import os
import sys
import time
from shutil import copy
from datetime import datetime
import operator
import argparse

#=== CONSTANTS ===========================================================

# Directories
BASE_SAVE_DIRECTORY = "/media/save/"

# Path to the file which keeps track of the last transfer date
TRANSFER_DATE_FILE_PATH = "/opt/rt3d/etc/DataCollector.txt"

# Base name of the folder to be created in the copy destination
DATA_COLLECTOR_FOLDER_NAME = "DataCollector"

# List of all acceptable extensions
ACCEPTED_EXTENSIONS = [".h5", ".dat"]

# Characters not allowed in directory names
PERIOD_CHAR = '.'

BYTES_TO_GB = 10**(-9)

#=== FUNCTIONS ===========================================================

def parseInputs(args):

    parser = argparse.ArgumentParser()
    parser.add_argument("outputDirectory" , help="The path to the output device")
    parser.add_argument("-a", "--all", help="If this flag is specified, all files will be copied regardless of creation date.", action="store_true")

    parsedList = parser.parse_args(args).__dict__

    if (not os.path.isdir(parsedList['outputDirectory'])):
        print("ERROR: Device path does not exist.")
        exit()

    return parsedList

def getLastDataTransfer(path):
    lastDataTransfer = 0.0

    # if it exists, read it.
    if os.path.isfile(path):
        try:
            with open(path, 'r') as f:
                allLines = f.readlines()
        except Exception as e:
            print("Could not open file: " + path)
            exit()

        # Go through all lines and save the latest valid as the return value
        for line in allLines:
            if isValidTime(line):
                lastDataTransfer = float(line)

    return lastDataTransfer

def isValidTime(string):
    try:
        float(string)
        return True
    except:
        return False

def setLastDataTransfer(path, newTime):
    with open(path, 'a') as file:
        file.write(str(newTime) + '\n')

def getFilesToBeTransferred(basepath, lastTransferDate):
    filesToBeTransferred = []
    for f in os.listdir(basepath):
        filepath = os.path.join(basepath,f)
        if os.path.isfile(filepath):
            name, ext = os.path.splitext(filepath)
            if ext in ACCEPTED_EXTENSIONS:
                if fileCreatedAfter(filepath, lastTransferDate):
                    filesToBeTransferred.append(filepath)
        # if it's a directory, go recursive
        elif os.path.isdir(filepath):
            filesToBeTransferred += getFilesToBeTransferred(filepath, lastTransferDate)
    return filesToBeTransferred

def sortByTimestamp(pathlist):
    timestampDict = {}
    for path in pathlist:
        timestampDict[path] = os.path.getctime(path)

    return sorted(timestampDict.items(), key=operator.itemgetter(1))

def fileCreatedAfter(filepath, date):
    return date < os.path.getctime(filepath)

def getFileSpace(pathList):
    totalSpace = 0
    for path in pathList:
        totalSpace += os.path.getsize(path)
    return totalSpace

def getFreeSpace(devicePath):
    info = os.statvfs(devicePath)
    return info.f_bsize * info.f_bfree

def createDirectory(path, newfolder):
    outPath = os.path.join(path, newfolder)
    try:
        os.mkdir(outPath)
    except Exception as e:
        print("Unable to create directory: " + outPath + ", " + str(e))
        exit()
    return outPath

def copyAllFiles(targetFiles, dest):

    filesCopied = 0
    for target in targetFiles:
        print("Copying file " + str(filesCopied+1) + " of " + str(len(targetFiles)))
        try:
            copy(target, dest)
            filesCopied+=1
        except Exception as e:
            print("Unable to copy file: " + target + ": " + str(e))
            break
    return filesCopied

#=== Main ================================================================
def main():

    # Parse inputs, get the device path
    parsedList = parseInputs(sys.argv[1:])
    outDevicePath = parsedList['outputDirectory']

    # Find the time of the last data transfer
    lastDataTransfer = getLastDataTransfer(TRANSFER_DATE_FILE_PATH)
    if(parsedList['all']):
        lastDataTransfer = 0

    # Get a list of all file paths to be copied
    allTransferFilePaths = getFilesToBeTransferred(BASE_SAVE_DIRECTORY, lastDataTransfer)

    # Sort the list by timestamp
    sortedFilePaths = sortByTimestamp(allTransferFilePaths)

    # Calculate how much space the files take up
    totalFileSpace = getFileSpace(allTransferFilePaths)
    print("Found " + str(len(allTransferFilePaths)) + " files totalling " + str(totalFileSpace * BYTES_TO_GB) + " GB.")
    if ( len(allTransferFilePaths) == 0):
        print("No files to transfer. Exiting...")
        exit()

    # Find out how much space is available on the drive
    freeSpace = getFreeSpace(outDevicePath)
    print("Free space on " + outDevicePath + ": " + str(freeSpace * BYTES_TO_GB) + " GB.")

    # Check to make sure there is enough space to copy the files
    if(totalFileSpace > freeSpace):
        print("WARNING: Not enough space on drive. Not all files will be copied.")
        print("\tTotal size of files: " + str(totalFileSpace))
        print("\tSpace available on device: " + str(freeSpace))

    # Generate the name of the save folder
    saveFolderName = DATA_COLLECTOR_FOLDER_NAME + "-" + str(datetime.now().month) + PERIOD_CHAR + str(datetime.now().day) + PERIOD_CHAR + str(datetime.now().year) + PERIOD_CHAR + str(time.time())

    # Create directories within the device to store the files
    saveDir = createDirectory(outDevicePath, saveFolderName)

    # Then copy the files
    filesCopied = copyAllFiles([path[0] for path in sortedFilePaths], saveDir)

    # update the last transfer file
    # if we made it all the way through the list, use now as the time. otherwise, use the timestamp of the last copied file.
    if(len(allTransferFilePaths) == filesCopied):
        setLastDataTransfer(TRANSFER_DATE_FILE_PATH, time.time())
    else:
        print("WARNING: Only copied " + str(filesCopied) + " of " + str(len(allTransferFilePaths)) + " files.")
        setLastDataTransfer(TRANSFER_DATE_FILE_PATH, sortedFilePaths[filesCopied - 1][1])


if __name__ == "__main__":
    main()
