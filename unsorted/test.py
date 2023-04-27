from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K
import random
import sys
import subprocess
import json
import dicom
import nrrd
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
img_rows = 256
img_cols = 256



# helper functions

# euclidean loss function
def euclidean_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred-y_true))

# weighteded euclidean loss function
def euclidean_loss_weighted(y_true, y_pred):
    return tf.reduce_mean(tf.multiply(tf.square(y_pred-y_true),y_true*255))

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
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.99, decay=0.0, nesterov=True), loss=euclidean_loss)
    return model


# train the network
def train(imgs_train, imgs_mask_train):

    # normalize the image data
    imgs_train = imgs_train.astype('float32')
    mean_image = np.mean(imgs_train)  # mean for data centering
    std_image = np.std(imgs_train)  # std for data normalization
    imgs_train -= mean_image
    imgs_train /= std_image

    # normalize the image label data
    imgs_mask_train = imgs_mask_train.astype('float32')
    mean_mask = np.mean(imgs_mask_train)  # mean for data centering
    std_mask = np.std(imgs_mask_train)  # std for data normalization
    imgs_mask_train -= mean_mask
    imgs_mask_train /= std_mask
    
    # print the model architecture
    model_checkpoint = ModelCheckpoint(filepath="checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss',save_best_only=True)

    # fit the model save the weights
    model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=1, verbose=1, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint])

    
if __name__ == '__main__':

        #train(gtVolume, trainVolume)
	# Parameters
	nextToken = 'eyJDb250aW51YXRpb25Ub2tlbiI6IG51bGwsICJib3RvX3RydW5jYXRlX2Ftb3VudCI6IDV9'
	batchSize = 1

	if(len(sys.argv) > 1):
		nextToken = str(sys.argv[1])

	# Process batches
	while(True):
		# List objects in current batch
		listingProc = subprocess.Popen(['aws', 's3api', 'list-objects-v2', '--bucket', 'nviewdatasets', '--starting-token', nextToken, '--max-items', str(batchSize), '--prefix', 'CT_datasets/', '--query', '[Contents[].{Key: Key}, NextToken]'],stdout=subprocess.PIPE)
		std_out, std_error = listingProc.communicate()
		data = json.loads(std_out)
		keys = data[0]
		# print keys[0]['Key'] test on how to use keys
		nextToken = data[1]
		
	      
		# Download the batch
		for k in range(0, len(keys)):
		    key = keys[k]['Key']
		    basekey = key.replace("CT_datasets/", "").replace(".nrrd", "")
		    gt = basekey + ".truth.nrrd"
		    train = basekey + ".train.nrrd"
		    reconKey = "Recons/recon" + basekey + ".nrrd"
		    print reconKey
		    subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'nviewdatasets', '--key',  key, gt])
		    subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'nviewdatasets', '--key',  reconKey, train])

		    # read the volumes
		    gtVolume, optionsGT = nrrd.read(gt) 
		    trainVolume, optionsTrain = nrrd.read(train)
		    print (gtVolume.shape)
		  

		    # train the network
		    train()
