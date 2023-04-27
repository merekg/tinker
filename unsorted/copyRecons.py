import subprocess
import json
import StringIO




def main():

    startIndex = 8500
    endIndex = 9833
    
    studyBucket = "TestSet_PredictedRecons"
    #objectlistCMD = "aws s3api list-objects-v2 --bucket nviewdatasets --prefix " + studyBucket + " --query '[Contents[].{Key: Key}]'"
    count = startIndex

    # list all objects in studyBucket
    objectlistCMD = "aws s3api list-objects-v2 --bucket nviewdatasets --prefix " + studyBucket + " --query '[Contents[].{Key: Key}]'"

    objectsInStudy = subprocess.Popen(objectlistCMD, shell=True, stdout=subprocess.PIPE)
    std_out, std_error = objectsInStudy.communicate()
    data = json.loads(std_out)

    while (count < endIndex):

        name = data[0][count]['Key']
        name = name.replace("TestSet_PredictedRecons/", "").replace("_Pred.nrrd", ".nrrd")

        fromBucket = 's3://nviewdatasets/Recons/' + name
        toBucket   = 's3://nviewdatasets/TestSet_InitialRecons/' + name
        print(fromBucket)
        print subprocess.check_output('aws s3 cp ' + fromBucket + ' ' + toBucket, shell=True)
        print("Count: {} File Name: {}".format(count, name))

        count += 1

    print("Complete.")

if __name__ == "__main__" :
    main()

        

