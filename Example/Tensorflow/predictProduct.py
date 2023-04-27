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
    predictions = np.swapaxes(predictions, 1, 2)[...,::-1]

    predictions = predictions/1.72
    # Write to nrrd 
    options={}
    spacing = 450.0/img_rows
    options['spacings'] = [spacing,spacing, spacing]
    options['keyvaluepairs'] = {}
    options['keyvaluepairs']['scale'] = '8192'
    
    predictions = np.clip(predictions, 0.0, float(2**16-1)).astype(np.uint16)
    nrrd.write(predictFileName, predictions,options)
    #nrrd.write(predictFileName.replace(".nrrd","_2.nrrd"), predictions,options)
    

# euclidean loss function
def euclidean_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred-y_true))


# weighteded euclidean loss function
def euclidean_loss_weighted(y_true, y_pred):
    return tf.reduce_mean(tf.multiply(tf.square(y_pred-y_true),y_true*255))


def standardizeVolume(Volume):

    # Swap the axes of the volumetric data to enable slicing on the axial data
    Volume = np.swapaxes(Volume, 1, 2)[...,::-1,:]

    # Finding three central slices to represent the volumetric information
    #Slices = Volume[96:161:32,:, :].astype(np.float32)
    Slices = Volume[:,:,:].astype(np.float32)
    
    # normalize the image data
    Slices -= 0
    Slices /= 4000.0
    Slices = Slices[..., np.newaxis]

    return Slices


def transformVolume(name):
    
    #subprocess.check_output(['teem-unu', 'swap', '-a',  str(0), str(2), '-i', name, '-o', name])
    #subprocess.check_output(['teem-unu', 'swap', '-a',  str(1), str(0), '-i', name, '-o', name])
    #subprocess.check_output(['teem-unu', 'flip', '-a',  str(0), '-i', name, '-o', name])
    #subprocess.check_output(['teem-unu', 'flip', '-a',  str(1), '-i', name, '-o', name])
    #subprocess.check_output(['teem-unu', 'flip', '-a',  str(2), '-i', name, '-o', name])
    
    
    Volume, options = nrrd.read(name)

    Volume = Volume*1.72
    # Write to nrrd 
    options={}
    options['spacings'] = [1.7578125,1.7578125, 1.7578125]
    options['keyvaluepairs'] = {}
    options['keyvaluepairs']['scale'] = '8192'

    #Volume = np.clip(Volume, 0.0, float(2**16-1)).astype(np.uint16)
    #nrrd.write(name.replace(".nrrd","_1.nrrd"), Volume,options)
    nrrd.write(name, Volume,options)

def revertVolumeTransformations(name):
    
    #subprocess.check_output(['teem-unu', 'flip', '-a',  str(2), '-i', name, '-o', name])
    #subprocess.check_output(['teem-unu', 'flip', '-a',  str(1), '-i', name, '-o', name])
    #subprocess.check_output(['teem-unu', 'flip', '-a',  str(0), '-i', name, '-o', name])
    #subprocess.check_output(['teem-unu', 'swap', '-a',  str(1), str(0), '-i', name, '-o', name])
    #subprocess.check_output(['teem-unu', 'swap', '-a',  str(0), str(2), '-i', name, '-o', name])
    
    Volume, options = nrrd.read(name)
   
    #Volume = Volume/2
    # Write to nrrd 
    options={}
    options['spacings'] = [1.7578125,1.7578125, 1.7578125]
    options['keyvaluepairs'] = {}
    options['keyvaluepairs']['scale'] = '8192'

    nrrd.write(name, Volume,options)
    #nrrd.write(name.replace(".nrrd","_3.nrrd"), Volume,options)

def readTestImageFromLocal(name):

    
    # Read the test image
    testImageReconVolume, optionsTestImageRecon = nrrd.read(name)
       
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
    
    predictFileName = str(sys.argv[1])
    modelName = str(sys.argv[2])
    model = readModel(modelName)

    # Transform and scale to fit in the same space as the training data
    # Not necessay if ML filter is trained on Rando or phantoms
    transformVolume(predictFileName)
    
    predictReconVolume = readTestImageFromLocal(predictFileName)
    predictAndSaveLocal(model, predictReconVolume,predictFileName)
    
    # Transform and scale to fit in the same space as the reconstruction engine data
    # Not necessay if ML filter is trained on Rando or phantoms
    #revertVolumeTransformations(predictFileName)
    


if __name__=="__main__":
    main()







