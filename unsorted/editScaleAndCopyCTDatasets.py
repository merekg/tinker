import subprocess
import json
import nrrd
import os



def main():


    finished = False
    studyBucket = "TestSet_PredictedRecons"
    #objectlistCMD = "aws s3api list-objects-v2 --bucket nviewdatasets --prefix " + studyBucket + " --query '[Contents[].{Key: Key}]'"
    count = 1

    # list all objects in studyBucket
    objectlistCMD = "aws s3api list-objects-v2 --bucket nviewdatasets --prefix " + studyBucket + " --query '[Contents[].{Key: Key}]'"

    objectsInStudy = subprocess.Popen(objectlistCMD, shell=True, stdout=subprocess.PIPE)
    std_out, std_error = objectsInStudy.communicate()
    data = json.loads(std_out)

    while (not finished):

        name = data[0][count]['Key']
        name = name.replace("TestSet_PredictedRecons/", "").replace("_Pred.nrrd", ".nrrd").replace("recon","")
        print("Count: {} File Name: {}".format(count, name))

        fromBucket = 's3://nviewdatasets/CT_datasets/' + name
        toBucket   = 's3://nviewdatasets/TestSet_GroundTruth/' + name
        subprocess.check_output('aws s3 cp ' + fromBucket + ' ' + name, shell=True)

        volume, options = nrrd.read(name)
        options['keyvaluepairs']['scale'] = '8192'
        nrrd.write(name , volume, options)

        subprocess.check_output('aws s3 cp ' + name + ' ' + toBucket, shell=True)

        os.remove(name)
        count += 1


        if (count > 1333):
            finished = True

    print("Complete.")

if __name__ == "__main__" :
    main()

        

