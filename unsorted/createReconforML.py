import subprocess
import os
import sys
import json
import logging
import traceback
import StringIO
import os.path


def AcqandRecon(seriesName, tempFolder):

    try:
        #singleInitial= str(sys.argv[1])
        if(os.path.exists("/dev/shm/nVReconstruction")):
            os.remove("/dev/shm/nVReconstruction")
        HOME = os.path.expanduser('~')
        #studyname = study.replace("/\n", "")
        #print("Study name: "+ studyname)
        studyBucket = 's3://nviewdatasets/CT_datasets/'

        #TODO: Add _recon at the end
        reconName = "recon" + seriesName


        objectKey = studyBucket + seriesName

        #Fetch Ground Truth from S3
        subprocess.check_output(['aws', 's3', 'cp', objectKey, tempFolder])

        print ("Folder:" +tempFolder+seriesName)

        # TODO: Need to update the config file for every study
        # Launch the Reconstruction and Acquisition
        reconProcess = subprocess.Popen([HOME+'/rt3d-build/bin/Reconstruction', '-v', '1', '-c', HOME+'/rt3d/ML/ReconSim.conf', '-q'])
        print("File: {} Reconstruction Done".format(seriesName))

        acqMode = open(HOME+"/rt3d/ML/inputForAcquisition.txt")
        print(tempFolder + seriesName)
        subprocess.Popen([HOME+'/rt3d-build/bin/AcquisitionSimulation', '-v', '1', '-c', HOME+'/rt3d/ML/AcqSim.conf', '-i', tempFolder + seriesName, '-q'],stdin = acqMode)
        reconProcess.communicate()
        print("File: {} Acquisition Done".format(seriesName))


        # rename the file at the default saved location
        subprocess.check_output(['mv', '/media/ramdisk/recon0.nrrd', reconName])

        # quantize the output volume (8-bit)
        subprocess.check_output("teem-unu quantize -b 16 -min 0 -max {} -i {} | teem-unu convert -t uint16 -o {}".format(float(2**16-1),reconName,reconName), shell=True)

        # upload the series converted .nrrd format on the cloud
        subprocess.check_output(['aws', 's3', 'cp', reconName,'s3://nviewdatasets/Recon_New/'])
        print("File: {} Uploading Recon file Done".format(seriesName))

        # Remove the input and the reconstructed volumes

        os.remove(tempFolder+seriesName)
        os.remove(reconName)

    except Exception as e:

        # Catch the exception and log the stack trace
        logging.error("----------------------------------------------------------------------------------------------------------------")
        exc_buffer = StringIO.StringIO()
        traceback.print_exc(file=exc_buffer)
        logging.error('File: ' + seriesName + ': Uncaught exception in worker process:\n%s',exc_buffer.getvalue())
        logging.error("----------------------------------------------------------------------------------------------------------------")
        pass

def main():


    # temp output folder to store Ground Truth
    tempFolder = str(sys.argv[1])
    finished = False
    studyBucket = "CT_datasets"
    #objectlistCMD = "aws s3api list-objects-v2 --bucket nviewdatasets --prefix " + studyBucket + " --query '[Contents[].{Key: Key}]'"
    
    startIndex = 0
    endIndex = 5
    count = startIndex


    while (count < endIndex):

        #list all objects in studyBucket
        objectlistCMD = "aws s3api list-objects-v2 --bucket nviewdatasets --prefix " + studyBucket + " --query '[Contents[].{Key: Key}]'"

        objectsInStudy = subprocess.Popen(objectlistCMD, shell=True, stdout=subprocess.PIPE)
        std_out, std_error = objectsInStudy.communicate()
        data = json.loads(std_out)
        print (len(data[0]))
        name = data[0][count]['Key']
        name = name.replace("CT_datasets/", "")
        print("Acquisition and Reconstruction process start for File: {}".format(name))

        count += 1
        AcqandRecon(name, tempFolder)


    print("Complete.")

if __name__ == "__main__" :
    main()

        

