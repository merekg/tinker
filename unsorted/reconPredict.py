from __future__ import print_function
import os
import string
import numpy as np
import json
import dicom
import nrrd
import sys
import gc
import subprocess
import logging
import StringIO
import traceback

def AcqandReconAws(fileName):

    try:
        #singleInitial= str(sys.argv[1])
        if(os.path.exists("/dev/shm/nVReconstruction")):
            os.remove("/dev/shm/nVReconstruction")
        HOME = os.path.expanduser('~')
        

        #TODO: Add _recon at the end
        seedName = fileName
        initialFileName = fileName.replace("recon","").replace("_Pred.nrrd",".nrrd")
        finalReconName = initialFileName.replace(".nrrd","_finalRecon.nrrd")

        objectKey = "CT_datasets/"+initialFileName
        seedKey = "TestSet_PredictedRecons/"+seedName


        #Fetch Ground Truth,Seed from S3
        subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'nviewdatasets', '--key', objectKey, initialFileName])
        subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'nviewdatasets', '--key', seedKey, seedName])
        
        # TODO: Need to update the config file for every study
        # Launch the Reconstruction and Acquisition
        reconProcess = subprocess.Popen([HOME+'/rt3d-build/bin/Reconstruction', '-v', '0', '-l', '-c', HOME+'/rt3d/ML/ReconSim.conf','-s', seedName, '-q'])

        acqMode = open(HOME+"/rt3d/ML/inputForAcquisition.txt")
        subprocess.Popen([HOME+'/rt3d-build/bin/AcquisitionSimulation', '-v', '0','-c', HOME+'/rt3d/ML/AcqSim.conf', '-i',  initialFileName, '-q'],stdin = acqMode)

        reconProcess.communicate()
        

        # rename the file at the default saved location
        subprocess.check_output(['mv', '/media/ramdisk/recon0.nrrd', finalReconName])

        # quantize the output volume (8-bit)
        subprocess.check_output("teem-unu quantize -b 16 -min 0 -max {} -i {} | teem-unu convert -t uint16 -o {}".format(float(2**16-1),finalReconName,finalReconName), shell=True)
        
         # upload the series converted .nrrd format on the cloud
        subprocess.check_output(['aws', 's3', 'cp', finalReconName,'s3://nviewdatasets/TestSet_SeededRecons/'])
        print("File: {} Uploading Recon file Done".format(finalReconName))

        # Remove the GT, seed and final recon files
        os.remove(seedName)
        os.remove(initialFileName)
        os.remove(finalReconName)

    except Exception as e:

        # Catch the exception and log the stack trace
        logging.error("----------------------------------------------------------------------------------------------------------------")
        exc_buffer = StringIO.StringIO()
        traceback.print_exc(file=exc_buffer)
        logging.error('File: ' + fileName + ': Uncaught exception in worker process:\n%s',exc_buffer.getvalue())
        logging.error("----------------------------------------------------------------------------------------------------------------")
        pass

def AcqandReconLocal(initialFileName, seedName):

    try:
        #singleInitial= str(sys.argv[1])
        if(os.path.exists("/dev/shm/nVReconstruction")):
            os.remove("/dev/shm/nVReconstruction")
        HOME = os.path.expanduser('~')
        

        #TODO: Add _recon at the end
        reconFinalName = initialFileName.replace(".nrrd","_finalRecon.nrrd")


        # TODO: Need to update the config file for every study
        # Launch the Reconstruction and Acquisition
        reconProcess = subprocess.Popen([HOME+'/rt3d-build/bin/Reconstruction', '-v', '0', '-l', '-c', HOME+'/rt3d/ML/ReconSim.conf','-s', seedName, '-q'])

        acqMode = open(HOME+"/rt3d/ML/inputForAcquisition.txt")
        subprocess.Popen([HOME+'/rt3d-build/bin/AcquisitionSimulation', '-v', '0','-c', HOME+'/rt3d/ML/AcqSim.conf', '-i',  initialFileName, '-q'],stdin = acqMode)

        reconProcess.communicate()
        

        # rename the file at the default saved location
        subprocess.check_output(['mv', '/media/ramdisk/recon0.nrrd', reconFinalName])

        # quantize the output volume (8-bit)
        subprocess.check_output("teem-unu quantize -b 16 -min 0 -max {} -i {} | teem-unu convert -t uint16 -o {}".format(float(2**16-1),reconFinalName,reconFinalName), shell=True)


    except Exception as e:

        # Catch the exception and log the stack trace
        logging.error("----------------------------------------------------------------------------------------------------------------")
        exc_buffer = StringIO.StringIO()
        traceback.print_exc(file=exc_buffer)
        logging.error('File: ' + initialFileName + ': Uncaught exception in worker process:\n%s',exc_buffer.getvalue())
        logging.error("----------------------------------------------------------------------------------------------------------------")
        pass    

def main() :
    
    startIndex = 1001
    endIndex = 1333
    
    if(len(sys.argv)==1) :
        studyBucket = "TestSet_PredictedRecons"
        count = startIndex

        while (count < endIndex):

            #list all objects in studyBucket
            objectlistCMD = "aws s3api list-objects-v2 --bucket nviewdatasets --prefix " + studyBucket + " --query '[Contents[].{Key: Key}]'"

            objectsInStudy = subprocess.Popen(objectlistCMD, shell=True, stdout=subprocess.PIPE)
            std_out, std_error = objectsInStudy.communicate()
            data = json.loads(std_out)
            predictFileName = data[0][count]['Key']
            predictFileName = predictFileName.replace("TestSet_PredictedRecons/", "")
            
            print("Count: {}; Predicting on File: {}".format(count, predictFileName))

            count += 1
            AcqandReconAws(predictFileName)
    else:
        initialFileName = str(sys.argv[1])
        seedName = str(sys.argv[2])
        AcqandReconLocal(initialFileName, seedName)
    

if __name__=="__main__":
    main()


