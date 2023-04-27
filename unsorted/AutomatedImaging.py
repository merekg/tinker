import subprocess
import os
import time
import h5py
from PIL import Image

HOME = os.path.expanduser('~')

INSTALLED_ACQUISITION_COMMAND = ["Acquisition", "--saveImages", "true", "--lowDose_kV", "72", "--lowDose_mA", "4"]
#INSTALLED_ACQUISITION_COMMAND = ["Acquisition", "--FileIO.SaveAcquisition=true", "--LowDose.kV=40", "--LowDose.mA=0.1"]
#BUILT_ACQUISITION_COMMAND = [HOME + "/rt3d-build/bin/Acquisition", "-c", HOME + "/rt3d/Apps/Acquisition/acquisition.conf", "--saveImages", "true"] 
#BUILT_ACQUISITION_COMMAND = [HOME + "/rt3d-build/bin/Acquisition", "-c", HOME + "/rt3d/Apps/Acquisition/acquisition.conf"] 

ACQUISITION_FILE_FOLDER = "/media/ramdisk/DefaultStudy/"
ACQUISITION_FILE_PATH = ACQUISITION_FILE_FOLDER + "DefaultStudy.i-.0.h5"
IMAGE_STACK_HDF5_PATH = "ITKImage/0/VoxelData"

START_DELAY_TIME = 60 # Seconds
ACQUISITION_RUN_TIME = 300 # Seconds
ACQUISITION_DELAY_TIME = 30 # Seconds
ACQUISITION_XRAY_TIME = 20 # Seconds
UNLOCK_WAIT_TIME = 1 # Seconds

def clearFolder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def main():

    # Clear out old acquisitions (if any)
    clearFolder(ACQUISITION_FILE_FOLDER)

    # Launch acquisition application
    print "Launching Acquisition..."
    acquisitionCommand = INSTALLED_ACQUISITION_COMMAND
    acquisitionProcess = subprocess.Popen(acquisitionCommand, stdin=subprocess.PIPE)

    # Wait for application warm-up time
    print "Waiting for " + str(START_DELAY_TIME) + " seconds..."
    time.sleep(START_DELAY_TIME)

    # Unlock x-ray
    print "Unlocking x-ray..."
    acquisitionProcess.stdin.write("x\n")
    time.sleep(UNLOCK_WAIT_TIME)

    # Start looping for acquisition run time
    startTime = time.time()
    while (time.time() < startTime + ACQUISITION_RUN_TIME):

        # Take an acquisition
        print "Tacking image acquisition..."
        acquisitionProcess.stdin.write("i-\n")

        # Wait for x-ray time
        time.sleep(ACQUISITION_XRAY_TIME)

        # Save out image capture from acquisition images
        with h5py.File(ACQUISITION_FILE_PATH, 'r') as acquisitionFile:
            image = Image.fromarray(acquisitionFile[IMAGE_STACK_HDF5_PATH][-1])
            image.save(str(int(time.time())) + ".png")

        # Delete old acquisition file
        clearFolder(ACQUISITION_FILE_FOLDER)

        # Wait for remainder of acquisition delay time
        time.sleep(ACQUISITION_DELAY_TIME - ACQUISITION_XRAY_TIME)

    # Wait in idle for user to stop
    key = raw_input("Press Enter to quit application...")
    
    # Quit acquisition application
    print "Quitting Acquisition..."
    acquisitionProcess.stdin.write("q\n")

if __name__ == "__main__" :
    main()
