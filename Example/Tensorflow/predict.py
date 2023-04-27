from __future__ import print_function
import os
import string
import tensorflow as tf
import numpy as np
import json
import dicom
import nrrd
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import backend as K
from sklearn import metrics
import keras.callbacks
import sys
import gc
import subprocess
import logging
import StringIO
import traceback
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
img_rows = 256
img_cols = 256


def predictAndSaveLocal(model, predictReconVolume, predictFileName):

    #model = readModel(modelName)
    
    predictions = model.predict(predictReconVolume, verbose=1)
    # Revert the transformations on the image before saving to nrrd file
    predictions *= 4000.0
    predictions += 0
    predictions = np.squeeze(predictions, axis=3)
    
    #Volume = np.swapaxes(Volume, 0, 2)
    predictions = np.swapaxes(predictions, 0, 2)

    # Write to nrrd 
    options={}
    spacing = 450.0/img_rows
    options['spacings'] = [spacing,spacing, spacing]
    options['keyvaluepairs'] = {}
    options['keyvaluepairs']['scale'] = '8192'
    
    predictions = np.clip(predictions, 0.0, float(2**16-1)).astype(np.uint16)
    nrrd.write(predictFileName+'_Pred.nrrd', predictions,options)
    print("The predicted file "+ predictFileName+"_Pred.nrrd is saved in your current directory")
    

def predictAndUploadAws(model, predictReconVolume, predictFileName):

    predictFileName = predictFileName.replace('.nrrd','_Pred.nrrd')

    #model = readModel(modelName)
    # Revert the transformations on the image before saving to nrrd file
    predictions = model.predict(predictReconVolume, verbose=1)

    predictions *= 4000.0
    predictions += 0
    predictions = np.squeeze(predictions, axis=3)
    
    #Volume = np.swapaxes(Volume, 0, 2)
    predictions = np.swapaxes(predictions, 0, 2)

    # Write to nrrd 
    options={}
    spacing = 450.0/img_rows
    options['spacings'] = [spacing,spacing, spacing]
    options['keyvaluepairs'] = {}
    options['keyvaluepairs']['scale'] = '8192'
    
    predictions = np.clip(predictions, 0.0, float(2**16-1)).astype(np.uint16)
    nrrd.write(predictFileName, predictions,options)
    
    # upload the series converted .nrrd format on the cloud
    subprocess.check_output(['aws', 's3', 'cp', predictFileName,'s3://companydatasets/TestSet_PredictedRecons/'])
    
    os.remove(predictFileName)
    #print("The predicted file "+ predictFileName+" has been uploaded to the TestSet_PredictedRecons folder on the cloud")
    

# euclidean loss function
def euclidean_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred-y_true))


# weighteded euclidean loss function
def euclidean_loss_weighted(y_true, y_pred):
    return tf.reduce_mean(tf.multiply(tf.square(y_pred-y_true),y_true*255))


def standardizeVolume(Volume):

    # Swap the axes of the volumetric data to enable slicing on the axial data
    Volume = np.swapaxes(Volume, 0, 2)

    # Finding three central slices to represent the volumetric information
    #Slices = Volume[96:161:32,:, :].astype(np.float32)
    Slices = Volume[:,:,:].astype(np.float32)
    
    # normalize the image data
    Slices -= 0
    Slices /= 4000.0
    Slices = Slices[..., np.newaxis]

    return Slices


def fetchTestImage(name):

    testImageReconKey = "Recons/"+name
    testImageRecon = name 

    # Fetch the test image
    subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'companydatasets', '--key', testImageReconKey,testImageRecon])

    # Read the test image
    testImageReconVolume, optionsTestImageRecon = nrrd.read(testImageRecon)

    # Precondition the data
    testImageReconVolume = standardizeVolume(testImageReconVolume)
    
    os.remove(testImageRecon)

    return testImageReconVolume


def readTestImageFromLocal(name):

    # Read the test image
    testImageReconVolume, optionsTestImageRecon = nrrd.read(name+".nrrd")
       
    # Precondition the data
    testImageReconVolume = standardizeVolume(testImageReconVolume)

    return testImageReconVolume


def readModel(modelName):
    
    # load json and create model
    json_file = open(modelName+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(modelName+".h5")

    # Compile the model before using it
    loaded_model.compile(optimizer=SGD(lr=1e-2, momentum=0.99, decay=0.0, nesterov=True), loss=euclidean_loss)
    print("Model loaded from specified file")

    return loaded_model


def main() :
    
    if(len(sys.argv)<2):
        print("The predictor expects path of the model concatenated with model name as the first argument.")
        quit()
    
    modelName = str(sys.argv[1])
    model = readModel(modelName)
    startIndex = 8501
    endIndex = 9833
    
    if(len(sys.argv)==2):

        studyBucket = "Recons"
        #objectlistCMD = "aws s3api list-objects-v2 --bucket companydatasets --prefix " + studyBucket + " --query '[Contents[].{Key: Key}]'"
        count = startIndex


        while (count < endIndex):

            #list all objects in studyBucket
            objectlistCMD = "aws s3api list-objects-v2 --bucket companydatasets --prefix " + studyBucket + " --query '[Contents[].{Key: Key}]'"

            objectsInStudy = subprocess.Popen(objectlistCMD, shell=True, stdout=subprocess.PIPE)
            std_out, std_error = objectsInStudy.communicate()
            data = json.loads(std_out)
            predictFileName = data[0][count]['Key']
            predictFileName = predictFileName.replace("Recons/", "")
            
            print("Count: {} Predicting on File: {}".format(count, predictFileName))

            count += 1
            predictReconVolume = fetchTestImage(predictFileName)
            predictAndUploadAws(model, predictReconVolume,predictFileName)

      
        
    if(len(sys.argv)==3):
        predictFileName = str(sys.argv[2])
        predictReconVolume = readTestImageFromLocal(predictFileName)
        predictAndSaveLocal(model, predictReconVolume,predictFileName)
    
    


if __name__=="__main__":
    main()

