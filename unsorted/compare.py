# import the necessary packages
#from skimage.measure import structural_similarity as ssim
from scipy import signal
#import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os
import nrrd
import json
import sys

    
def compare_images(imageA, imageB, title, file):
    # compute the mean squared error and structural similarity
    # index for the images

    HOME = os.path.expanduser('~')

    output = subprocess.check_output([HOME + '/rt3d-build/bin/Statistics', '--img1',imageA,'--img2',imageB], stderr=subprocess.STDOUT)
    output = [line for line in output.split("\n") if(line.startswith( "RMS: ")) ]
    print(output)
    file.write(title + ":"+output[0])
    file.write("\n")


def fetchVolumesAws(fileName, tempFolder):
    
    
    studyBuckets = {'gTStudyBucket' : "TestSet_GroundTruth",  'initialReconStudyBucket' : "TestSet_InitialRecons", 'predictedReconStudyBucket' : "TestSet_PredictedRecons", 'seededReconStudyBucket' : "TestSet_SeededRecons", 'secondIterationPredictedReconStudyBucket' : "TestSet_SecondIterationPredictedRecons"}
    volumeFilePaths = {}
    #volumes = {}
    
    for bucketName, bucketValue in studyBuckets.iteritems():
        fileNameBucket ={
                    'gTStudyBucket': lambda fileName : fileName.replace("recon", "").replace("_Pred", ""),
                    'initialReconStudyBucket': lambda fileName : fileName.replace("_Pred", ""),
                    'predictedReconStudyBucket': lambda fileName : fileName,
                    'seededReconStudyBucket': lambda fileName : fileName.replace("recon", "").replace("_Pred", "_finalRecon"),
                    'secondIterationPredictedReconStudyBucket': lambda fileName : fileName.replace("recon", "").replace("_Pred", "_finalRecon_Pred")
                }[bucketName](fileName)
        
        subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'nviewdatasets', '--key', bucketValue+ "/" + fileNameBucket, tempFolder+fileNameBucket])
        volumeFilePaths[bucketValue] = tempFolder+fileNameBucket
        #volumes[bucketValue], optionsGT = nrrd.read(fileNameBucket)
        #volumes[bucketValue] = np.swapaxes(volumes[bucketValue], 0, 2)
        #os.remove(fileNameBucket)

    return volumeFilePaths


def fetchProjectionsAws(fileName, tempFolder):
    studyBuckets = {'gTStudyBucket': "TestSet_GroundTruth_proj", 'initialReconStudyBucket': "TestSet_InitialRecons_proj",
                    'predictedReconStudyBucket': "TestSet_PredictedRecons_proj",
                    'seededReconStudyBucket': "TestSet_SeededRecons_proj",
                    'secondIterationPredictedReconStudyBucket': "TestSet_SecondIterationPredictedRecons_proj"}
    projFilePaths = {}
    # volumes = {}

    for bucketName, bucketValue in studyBuckets.iteritems():
        fileNameBucket = {
            'gTStudyBucket': lambda fileName: fileName.replace("recon", "").replace("_Pred", "_proj"),
            'initialReconStudyBucket': lambda fileName: fileName.replace("_Pred", "_proj"),
            'predictedReconStudyBucket': lambda fileName: fileName.replace(".nrrd","_proj.nrrd"),
            'seededReconStudyBucket': lambda fileName: fileName.replace("recon", "").replace("_Pred", "_finalRecon_proj"),
            'secondIterationPredictedReconStudyBucket': lambda fileName: fileName.replace("recon", "").replace("_Pred",
                                                                                                               "_finalRecon_Pred_proj")
        }[bucketName](fileName)

        subprocess.check_output(
            ['aws', 's3api', 'get-object', '--bucket', 'nviewdatasets', '--key', bucketValue + "/" + fileNameBucket,
             tempFolder + fileNameBucket])
        projFilePaths[bucketValue] = tempFolder + fileNameBucket
        # volumes[bucketValue], optionsGT = nrrd.read(fileNameBucket)
        # volumes[bucketValue] = np.swapaxes(volumes[bucketValue], 0, 2)
        # os.remove(fileNameBucket)

    return projFilePaths

def compareP(pathBucket,file):


    compare_images(pathBucket['TestSet_GroundTruth_proj'], pathBucket['TestSet_GroundTruth_proj'],
                   "Ground Truth Proj vs. Ground Truth Proj", file)

    compare_images(pathBucket['TestSet_GroundTruth_proj'], pathBucket['TestSet_InitialRecons_proj'], "Ground Truth Proj vs. Initial Recon Proj", file)
    compare_images(pathBucket['TestSet_GroundTruth_proj'], pathBucket['TestSet_PredictedRecons_proj'], "Ground Truth Proj vs. Predicted Recon Proj", file)
    compare_images(pathBucket['TestSet_GroundTruth_proj'], pathBucket['TestSet_SeededRecons_proj'], "Ground Truth Proj vs. Seeded Recon Proj", file)
    compare_images(pathBucket['TestSet_GroundTruth_proj'], pathBucket['TestSet_SecondIterationPredictedRecons_proj'], "Ground Truth Proj vs. Second Iteration Predicted Recon Proj", file)

    compare_images(pathBucket['TestSet_InitialRecons_proj'], pathBucket['TestSet_PredictedRecons_proj'], "Initial Recon Proj vs. Predicted Recon Proj", file)
    compare_images(pathBucket['TestSet_InitialRecons_proj'], pathBucket['TestSet_SeededRecons_proj'], "Initial Recon Proj vs. Seeded Recon Proj", file)
    compare_images(pathBucket['TestSet_InitialRecons_proj'], pathBucket['TestSet_SecondIterationPredictedRecons_proj'], "Initial Recon Proj vs. Second Iteration Predicted Recon Proj", file)

    compare_images(pathBucket['TestSet_PredictedRecons_proj'], pathBucket['TestSet_SeededRecons_proj'], "Predicted Recon Proj vs. Seeded Recon Proj", file)
    compare_images(pathBucket['TestSet_PredictedRecons_proj'], pathBucket['TestSet_SecondIterationPredictedRecons_proj'], "Predicted Recon Proj vs. Second Iteration Predicted Recon Proj", file)

    compare_images(pathBucket['TestSet_SeededRecons_proj'], pathBucket['TestSet_SecondIterationPredictedRecons_proj'], "Seeded Recon Proj vs. Second Iteration Predicted Recon Proj", file)


def compare(pathBucket,file):


    compare_images(pathBucket['TestSet_GroundTruth'], pathBucket['TestSet_GroundTruth'],
                   "Ground Truth vs. Ground Truth", file)

    compare_images(pathBucket['TestSet_GroundTruth'], pathBucket['TestSet_InitialRecons'], "Ground Truth vs. Initial Recon", file)
    compare_images(pathBucket['TestSet_GroundTruth'], pathBucket['TestSet_PredictedRecons'], "Ground Truth vs. Predicted Recon", file)
    compare_images(pathBucket['TestSet_GroundTruth'], pathBucket['TestSet_SeededRecons'], "Ground Truth vs. Seeded Recon", file)
    compare_images(pathBucket['TestSet_GroundTruth'], pathBucket['TestSet_SecondIterationPredictedRecons'], "Ground Truth vs. Second Iteration Predicted Recon", file)

    compare_images(pathBucket['TestSet_InitialRecons'], pathBucket['TestSet_PredictedRecons'], "Initial Recon vs. Predicted Recon", file)
    compare_images(pathBucket['TestSet_InitialRecons'], pathBucket['TestSet_SeededRecons'], "Initial Recon vs. Seeded Recon", file)
    compare_images(pathBucket['TestSet_InitialRecons'], pathBucket['TestSet_SecondIterationPredictedRecons'], "Initial Recon vs. Second Iteration Predicted Recon", file)

    compare_images(pathBucket['TestSet_PredictedRecons'], pathBucket['TestSet_SeededRecons'], "Predicted Recon vs. Seeded Recon", file)
    compare_images(pathBucket['TestSet_PredictedRecons'], pathBucket['TestSet_SecondIterationPredictedRecons'], "Predicted Recon vs. Second Iteration Predicted Recon", file)

    compare_images(pathBucket['TestSet_SeededRecons'], pathBucket['TestSet_SecondIterationPredictedRecons'], "Seeded Recon vs. Second Iteration Predicted Recon", file)


def main():

    tempFolder = sys.argv[1]
    studyBucket = "TestSet_PredictedRecons"
    startIndex = 1
    endIndex = 2
    count = startIndex
    
    file = open("quantitativeAnalysis.txt", "a")

    while (count < endIndex):

        #list all objects in studyBucket
        objectlistCMD = "aws s3api list-objects-v2 --bucket nviewdatasets --prefix " + studyBucket + " --query '[Contents[].{Key: Key}]'"

        objectsInStudy = subprocess.Popen(objectlistCMD, shell=True, stdout=subprocess.PIPE)
        std_out, std_error = objectsInStudy.communicate()
        data = json.loads(std_out)
        fileName = data[0][count]['Key']
        fileName = fileName.replace("TestSet_PredictedRecons/", "")
        file.write(" Count: {}; File Name: {}".format(count, fileName))
        file.write("\n")

        volumeFilePaths = fetchVolumesAws(fileName, tempFolder)
        projFilePaths = fetchProjectionsAws(fileName, tempFolder)

        compare(volumeFilePaths,file)
        compareP(projFilePaths,file)
        for name in volumeFilePaths.itervalues():
            os.remove(name)
        for name in projFilePaths.itervalues():
            os.remove(name)
 

        count += 1
    file.close()
         
def main1():
    image1 = sys.argv[1]
    image2 = sys.argv[2]

    file = open("quantitativeAnalysis.txt", "a")

    volume1, options = nrrd.read(image1)
    volume2, options = nrrd.read(image2)


    compare_images(volume1, volume2, "Image1 vs Image2", file)

    file.close()

if __name__=="__main__":
    main()
