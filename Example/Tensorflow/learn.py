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
import keras.callbacks
from matplotlib import pyplot as plt
from skimage.io import imsave
from keras.utils import plot_model
from keras import backend as K
from PIL import Image
import random
import sys
import subprocess
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
img_rows = 256
img_cols = 256

# Creates a session with log_device_placement set to True.
# This adds verbosity to see if we run in CPU or GPU
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

class prediction_history(keras.callbacks.Callback):

    def __init__(self):
        self.predhis = []

    def saveModel(self):

        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model_secondIteration.json", "w") as json_file:
            json_file.write(model_json)
        
        # serialize weights to HDF5
        self.model.save_weights("model_secondIteration.h5")
    

    def predictOnGoldenImages(self):
        global good_image_gtVolume,good_image_trainVolume
        global good_image_gtVolume_1,good_image_trainVolume_1

        #Run the keras predictor
        predictions = self.model.predict(good_image_trainVolume, verbose=0)
        predictions_1 = self.model.predict(good_image_trainVolume_1, verbose=0)

        # Revert the transformations on the golden image before saving to nrrd file
        predictions *= 4000.0
        predictions += 0
        predictions = np.swapaxes(predictions, 0, 2)
        predictions_1 *= 4000.0
        predictions_1 += 0
        predictions_1 = np.swapaxes(predictions_1, 0, 2)
        
        # Write to nrrd 
        options={}
        spacing = 450.0/img_rows
        zspacing = spacing * 32
        options["spacings"]=[spacing,spacing, zspacing]
        predictions = np.clip(predictions, 0.0, float(2**16-1)).astype(np.uint16)
        predictions_1 = np.clip(predictions_1, 0.0, float(2**16-1)).astype(np.uint16)
        nrrd.write('TestPred.nrrd', predictions, options)
        nrrd.write('TestPred_1.nrrd', predictions_1, options)


    def on_epoch_end(self, batch, logs={}):

        # Save the model
        self.saveModel()

        # Predict on the two golden images
        self.predictOnGoldenImages()
       

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
    Slices = Volume[96:161:32,:, :].astype(np.float32)
    
    # normalize the image data
    Slices -= 0
    Slices /= 4000.0
    Slices = Slices[..., np.newaxis]

    return Slices


def fetchTestImage(name):

    #reconPrefix = "Recons/recon"
    reconPrefix = "Recon_withMLSeed/"
    reconSuffix = "_finalRecon"
    #reconSuffix = ""
    
    good_image_gt_key = "CT_datasets/"+name+".nrrd"
    good_image_recon_key = reconPrefix+name+reconSuffix+".nrrd"
    good_image_truth = name + ".truth.nrrd"
    good_image_train = name + ".train.nrrd"

    # Fetch the two golden images
    subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'companydatasets', '--key', good_image_gt_key, good_image_truth])
    subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'companydatasets', '--key', good_image_recon_key,good_image_train])

    # Read the golden images
    good_image_gtVolume, options_good_image_gt = nrrd.read(good_image_truth)
    good_image_trainVolume, options_good_imageTrain = nrrd.read(good_image_train)
    
    # Delete the local copies after reading the nrrd files
    os.remove(good_image_truth)
    os.remove(good_image_train)
    
    # Precondition the data
    good_image_gtVolume = standardizeVolume(good_image_gtVolume)
    good_image_trainVolume = standardizeVolume(good_image_trainVolume)

    return good_image_gtVolume,good_image_trainVolume


# network similar to unet (4 layer UNET)
def get_net():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Dropout(0.5)(conv5)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1))(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=SGD(lr=1e-2, momentum=0.99, decay=0.0, nesterov=True), loss=euclidean_loss)
    return model


#firstFile = "Recons/recon1.2.392.200036.9116.2.2.2.1762579135.1471311146.25329.nrrd"
firstFile = "Recon_withMLSeed/1.2.392.200036.9116.2.2.2.1762579135.1471311146.25329_finalRecon.nrrd"
currentFile = firstFile

def generateValidateRecon(batchSize, validationSize):
    global firstFile

    #reconBucket = "Recons/"
    reconBucket = "Recon_withMLSeed/"
    
    #reconPrefix = "Recons/recon"
    reconPrefix = "Recon_withMLSeed/"
    reconSuffix = "_finalRecon"
    #reconSuffix = ""
    
    lastValidationFile = firstFile

    while (True):

        # list "BatchSize" number of objects in Recon dataset
        objectlistCMD = "aws s3api list-objects-v2 --bucket companydatasets --start-after " + lastValidationFile + " --max-items " + str(batchSize+validationSize) + " --prefix " + reconBucket +" --query '[Contents[].{Key: Key}]'"

        #Find the name of the file
        objectsInStudy = subprocess.Popen(objectlistCMD, shell=True, stdout=subprocess.PIPE)
        std_out, std_error = objectsInStudy.communicate()
        data = json.loads(std_out)
        if not data or len(data)==0:
            #TODO: possibility of an infinite loop here. add some sort of a sanity counter
            lastValidationFile = firstFile
            continue
        keys = data[0]
        if not keys:
            #print("No keys retrieved from data. tying again at first file")
            lastValidationFile = firstFile
            continue 
        lastValidationFile = keys[-1]['Key']
        if len(keys) < batchSize+validationSize:
            #print("retrieved number of keys is less than requested keys. wrapping around and retreiving "+str(batchSize+validationSize-len(keys))+" more file(s)")
            objectlistCMD = "aws s3api list-objects-v2 --bucket companydatasets  --max-items " + str(batchSize+validationSize-len(keys)) + " --prefix " + reconBucket +" --query '[Contents[].{Key: Key}]'"
            #Find the name of the file
            objectsInStudy = subprocess.Popen(objectlistCMD, shell=True, stdout=subprocess.PIPE)
            std_out, std_error = objectsInStudy.communicate()
            data = json.loads(std_out)
            if not data or len(data)==0:
                sys.exit(1)
            keys += data[0]
            lastValidationFile = keys[-1]['Key']

        #print("generateValidateRecon: Validation File Names:")
        #for k in range(batchSize, len(keys)) : print(str(keys[k]['Key']))

        gtBatch = np.empty([0,256,256,1], dtype=np.float32)
        reconBatch = np.empty([0,256,256,1], dtype=np.float32)

        for k in range(batchSize, len(keys)): #keys[0] = 1st Batch File, key[batchSize] = 1st Validation File

            reconKey = keys[k]['Key']
            currentFile = reconKey

            #print(' Batch is: {} Current Validation file is {}:'.format(batchSize,currentFile))
            basekey = reconKey.replace(reconPrefix, "").replace(reconSuffix, "")

            # Constructing recon and GT names
            gt = basekey.replace(".nrrd", ".truth.nrrd")
            recon = basekey.replace(".nrrd", ".recon.nrrd")
            gtkey = "CT_datasets/" + basekey

            # Fetch the files from aws
            subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'companydatasets', '--key', gtkey, gt])
            subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'companydatasets', '--key', reconKey, recon])

            # read the volumes and free-up disk space
            gtVolume, optionsGT = nrrd.read(gt)
            reconVolume, optionsTrain = nrrd.read(recon)
            os.remove(gt)
            os.remove(recon)

            #Precondition Inputs
            gtSlices = standardizeVolume(gtVolume)
            reconSlices = standardizeVolume(reconVolume)

            yield gtSlices, reconSlices


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


def generateGtReconPairs(batchSize, validationSize):
    global firstFile,good_image_name
    global currentFile
    lastValidationFile = currentFile
    #currentFile = firstFile
    count = 0
    #reconBucket = "Recons/"
    reconBucket = "Recon_withMLSeed/"
    
    #reconPrefix = "Recons/recon"
    #reconSuffix = ""
    reconPrefix = "Recon_withMLSeed/"
    reconSuffix = "_finalRecon"
    
    
    while (True):
        
        count = count + batchSize + validationSize
        #rint("Count: {}".format(count))
        if (count > 8505):
            count = batchSize + validationSize
            lastValidationFile = firstFile        
            print("Finished iterating over training set. Starting at the first file: {}".format(lastValidationFile))

        #list "BatchSize" number of objects in Recon dataset
        objectlistCMD = "aws s3api list-objects-v2 --bucket companydatasets --start-after " + lastValidationFile + " --max-items " + str(batchSize+validationSize) + " --prefix " + reconBucket +" --query '[Contents[].{Key: Key}]'"

        #Find the name of the file
        objectsInStudy = subprocess.Popen(objectlistCMD, shell=True, stdout=subprocess.PIPE)
        std_out, std_error = objectsInStudy.communicate()
        data = json.loads(std_out)
        if (not data) or len(data)==0:
            #TODO: possibility of an infinite loop here. add some sort of a sanity counter
            print("No data read from S3. trying again at first File")
            lastValidationFile = firstFile
            continue
        keys = data[0]
        if not keys:
            print("No keys retrieved from data. tying again at first file")
            lastValidationFile = firstFile
            continue 
        lastValidationFile = keys[-1]['Key']
        if len(keys) < batchSize+validationSize:
            print("retrieved number of keys is less than requested keys. wrapping around and retreiving "+str(batchSize+validationSize-len(keys))+" more file(s)")
            objectlistCMD = "aws s3api list-objects-v2 --bucket companydatasets  --max-items " + str(batchSize+validationSize-len(keys)) + " --prefix " + reconBucket +" --query '[Contents[].{Key: Key}]'"
            #Find the name of the file
            objectsInStudy = subprocess.Popen(objectlistCMD, shell=True, stdout=subprocess.PIPE)
            std_out, std_error = objectsInStudy.communicate()
            data = json.loads(std_out)
            if not data or len(data)==0:
                sys.exit(1)
            keys += data[0]
            lastValidationFile = keys[-1]['Key']
            lvfLoopCounter = -2
            while((not lastValidationFile.endswith(".nrrd")) and lvfLoopCounter+len(keys)>0 ):
                lastValidationFile = keys[lvfLoopCounter]['Key']
                lvfLoopCounter -= 1
            
        #print("gtreconpairs: Validation File Names:")
        #for k in range(batchSize, len(keys)) : print(str(keys[k]['Key']))

        gtBatch = np.empty([0,256,256,1], dtype=np.float32)
        reconBatch = np.empty([0,256,256,1], dtype=np.float32)

        for k in range(0, batchSize):

            reconKey = keys[k]['Key']
            
            #basekey = "Recons/recon"+ good_image_name + ".nrrd"
            currentFile = reconKey

            #print(' Batch is: {} Current file is {}:'.format(batchSize,currentFile))
            basekey = reconKey.replace(reconPrefix, "").replace(reconSuffix, "")
            

            # Do not train on test image
            #if(basekey == good_image_name):
            #    print("Skipping training on Test Image")
            #    continue
            #print("Training on File: {}".format(basekey))

            # Constructing recon and GT names
            gt = basekey.replace(".nrrd", ".truth.nrrd")
            recon = basekey.replace(".nrrd", ".recon.nrrd")
            gtkey = "CT_datasets/" + basekey

            # Fetch the files from aws
            try :
                subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'companydatasets', '--key', gtkey, gt])
                subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'companydatasets', '--key', reconKey, recon])
            except :
                continue

            # read the volumes and free-up disk space
            gtVolume, optionsGT = nrrd.read(gt)
            reconVolume, optionsTrain = nrrd.read(recon)
            os.remove(gt)
            os.remove(recon)

            #Check for min and max Density values and discard if they fall out of range
            gtMinDensity = gtVolume.min()
            gtMaxDensity = gtVolume.max()
            reconMinDensity = np.min(reconVolume)
            reconMaxDensity = np.max(reconVolume)
           
            if(gtMaxDensity > 10000):
                #print("Discarding File:{}".format(currentFile))
                continue

            #Precondition Inputs
            gtSlices = standardizeVolume(gtVolume)
            reconSlices = standardizeVolume(reconVolume)

            #Add to batch
            gtBatch = np.concatenate((gtBatch,gtSlices),axis = 0)
            reconBatch = np.concatenate((reconBatch,reconSlices),axis = 0)

        print(' Batch of {} ,next starts at {}:'.format(batchSize, lastValidationFile))
        yield reconBatch, gtBatch


# Main script TODO create a main() function
batchSize = 8
validationSize = 1

# Find the length of the Recon folder to compute steps per epoch
#findLengthCMD = "aws s3api list-objects --bucket companydatasets --prefix Recons/ --output json --query 'length(Contents[])'"
#objects = subprocess.Popen(findLengthCMD, shell=True, stdout=subprocess.PIPE)
#std_out, std_error = objects.communicate()
#lengthRecon = json.loads(std_out)
#stepsPerEpoch = int(lengthRecon)/batchSize
#stepsPerEpoch = 8505/(batchSize+validationSize)
stepsPerEpoch = 8505/(batchSize+validationSize)

# If the model file is specified, read from the model. Otherwise, initialize the model for the first time
if(len(sys.argv) > 1):
    model = readModel(str(sys.argv[1]))
else:
    model = get_net()

# A "good" nrrd to test for quality of training
good_image_name = "1.2.392.200036.9116.2.2.2.1762579135.1471311146.25329"
good_image_gtVolume, good_image_trainVolume = fetchTestImage(good_image_name)
good_image_name_1 = "1.2.124.113532.12.10544.32487.20140731.81254.67829147"
good_image_gtVolume_1, good_image_trainVolume_1 = fetchTestImage(good_image_name_1)

# Callback to predict on the golden image and save the last model at the end of each epoch
history = prediction_history()

# Callback to save the model with the least loss
checkpoint_bestModel = keras.callbacks.ModelCheckpoint('bestModel_secondIteration.hdf5', monitor='loss', verbose=0, save_best_only=True, mode='min')
#model.fit_generator(generateGtReconPairs(batchSize, validationSize),steps_per_epoch=1200/batchSize, validation_data=generateValidateRecon(batchSize, validationSize), validation_steps= 1, epochs=10, max_queue_size=1,verbose=1,callbacks=[history])

# Call the fitter
model.fit_generator(generateGtReconPairs(batchSize, validationSize),steps_per_epoch=stepsPerEpoch, epochs=15000, validation_data=generateValidateRecon(batchSize, validationSize), validation_steps= 1, max_queue_size=1,verbose=1,callbacks=[history,checkpoint_bestModel])

