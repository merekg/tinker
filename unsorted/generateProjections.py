import subprocess
import os
import sys
import json
import logging
import traceback
import StringIO
import os.path


def AcqSim(seriesName, studyBucket):

    try:
        #singleInitial= str(sys.argv[1])
        if(os.path.exists("/dev/shm/nVReconstruction")):
            os.remove("/dev/shm/nVReconstruction")
        HOME = os.path.expanduser('~')
        #studyname = study.replace("/\n", "")
        #print("Study name: "+ studyname)

        destBucket = 's3://nviewdatasets/' + studyBucket+'_proj/'
        studyBucket = 's3://nviewdatasets/' + studyBucket+'/'

        projName = seriesName.replace(".nrrd", "_proj.nrrd")


        objectKey = studyBucket + seriesName

        #Fetch Ground Truth from S3
        subprocess.check_output(['aws s3 cp '+objectKey+' '+seriesName],shell=True)

        # TODO: Need to update the config file for every study
        # Launch Acquisition Simulation

        acqMode = open(HOME+"/rt3d/ML/inputForAcquisition.txt")
        simulationProcess = subprocess.Popen([HOME+'/rt3d-build/bin/AcquisitionSimulation', '-v', '0', '-c', HOME+'/rt3d/ML/AcqSim.conf', '-i', seriesName, '-q'],stdin = acqMode)
        simulationProcess.communicate()

        # quantize the output volume (8-bit)
        #subprocess.check_output("teem-unu quantize -b 16 -min 0 -max {} -i {} | teem-unu convert -t uint16 -o {}".format(float(2**16-1),reconName,reconName), shell=True)

        # upload the series converted .nrrd format on the cloud
        subprocess.check_output(['aws', 's3', 'cp', projName,destBucket])
        print("File: {} Uploading Projection Done".format(seriesName))

        # Remove the input and the reconstructed volumes

        os.remove(seriesName)
        os.remove(projName)

    except Exception as e:

        # Catch the exception and log the stack trace
        logging.error("----------------------------------------------------------------------------------------------------------------")
        exc_buffer = StringIO.StringIO()
        traceback.print_exc(file=exc_buffer)
        logging.error('File: ' + seriesName + ': Uncaught exception in worker process:\n%s',exc_buffer.getvalue())
        logging.error("----------------------------------------------------------------------------------------------------------------")
        pass

def main():


    finished = False
    if(len(sys.argv) != 2):
        print("Enter the study bucket to generate projections in as an input")
        exit(1)
    studyBucket = sys.argv[1]
    #objectlistCMD = "aws s3api list-objects-v2 --bucket nviewdatasets --prefix " + studyBucket + " --query '[Contents[].{Key: Key}]'"
    count = 1


    while (not finished):

        #list all objects in studyBucket
        objectlistCMD = "aws s3api list-objects-v2 --bucket nviewdatasets --prefix " + studyBucket + " --query '[Contents[].{Key: Key}]'"

        objectsInStudy = subprocess.Popen(objectlistCMD, shell=True, stdout=subprocess.PIPE)
        std_out, std_error = objectsInStudy.communicate()
        data = json.loads(std_out)
        print (len(data[0]))
        name = data[0][count]['Key']
        name = name.replace(studyBucket+"/", "")
        print("Count: {} File Name: {} Study Name: {}".format(count, name, studyBucket))

        count += 1
        AcqSim(name, studyBucket)

        if (count > len(data[0])):
            finished = True

    print("Complete.")

if __name__ == "__main__" :
    main()

        

