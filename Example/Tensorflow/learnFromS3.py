from __future__ import print_function
import os
import string
import tensorflow as tf
import numpy as np
import json
import dicom
import nrrd
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
from matplotlib import pyplot as plt
from skimage.io import imsave
from PIL import Image
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
img_rows = 256
img_cols = 256

# Creates a session with log_device_placement set to True.
# This adds verbosity to see if we run in CPU or GPU
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

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

def draw_image(imgs_mask_pred, imgs, imgs_mask, displayname):

    pred_dir = '../ML/MLresults/preds'
    gt_dir = '../ML/MLresults/gt'
    input_dir = '../ML/MLresults/input'
    ctr = 0
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(gt_dir):
        os.mkdir(gt_dir)
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    for image, image_id, input_image in zip(imgs_mask_pred, imgs, imgs_mask):
        #image -= image.min()
        #image /= image.max()
        #image = (image[:, :, 0] * 65536.).astype(np.uint16)
        image = image[:, :, 0].astype(np.uint16)

        #image_id -= image_id.min()
        #image_id /= image_id.max()
        #image_id = (image_id[:, :, 0] * 65536.).astype(np.uint16)
        image_id = image_id[:, :, 0].astype(np.uint16)

        #input_image -= input_image.min()
        #input_image /= input_image.max()
        #input_image = (input_image[:, :, 0] * 65536.).astype(np.uint16)
        input_image = input_image[:, :, 0].astype(np.uint16)

        imsave(os.path.join(pred_dir, str(ctr) + '_pred.png'), image)
        imsave(os.path.join(gt_dir, str(ctr) + '_gt.png'), image_id)
        imsave(os.path.join(input_dir, str(ctr) + '_input.png'), input_image)
        ctr = ctr + 1

    nrows = 3
    fig, ax = plt.subplots(nrows=nrows, ncols=3, figsize=(12, 10))
    for row in range(nrows):
        ax[row][0].imshow(np.squeeze(imgs[row * 10, :, :, 0]), vmin = 0, vmax = 2000, cmap='gray')
        ax[row][1].imshow(np.squeeze(imgs_mask[row * 10, :, :, 0]), vmin = 0, vmax = 2000, cmap='gray')
        ax[row][2].imshow(np.squeeze(imgs_mask_pred[row * 10, :, :, 0]), vmin = 0, vmax = 2000, cmap='gray')

    for a in ax.flatten():
        a.grid(False)
        a.axis('off')

    ax[0][0].set_title('Ground truth')
    ax[0][1].set_title('Input')
    ax[0][2].set_title('Prediction')
    fig.tight_layout()
    fig.savefig("../ML/MLresults/"+displayname+".png")


def validate_and_predict(model, imgs_valid_predict, imgs_mask_valid_predict, displayname):

    # make 4-D array
    imgs_valid_predict = imgs_valid_predict[..., np.newaxis]
    imgs_mask_valid_predict = imgs_mask_valid_predict[..., np.newaxis]

    # normalize the image data
    imgs_valid_predict = imgs_valid_predict.astype('float32')
    #mean_image = np.mean(imgs_valid_predict)  # mean for data centering
    #std_image = np.std(imgs_valid_predict)  # std for data normalization
    #imgs_valid_predict -= mean_image
    #imgs_valid_predict /= std_image
    imgs_valid_predict -= 1000
    imgs_valid_predict /= 2000

    # normalize the image label data
    imgs_mask_valid_predict = imgs_mask_valid_predict.astype('float32')
    #mean_mask = np.mean(imgs_mask_valid_predict)  # mean for data centering
    #std_mask = np.std(imgs_mask_valid_predict)  # std for data normalization
    #imgs_mask_valid_predict -= mean_mask
    #imgs_mask_valid_predict /= std_mask
    imgs_mask_valid_predict -= 1000
    imgs_mask_valid_predict /= 2000

    # Evaluate the model
    loss = model.evaluate(imgs_valid_predict, imgs_mask_valid_predict, verbose=0)

    # Use the model for prediction
    predictions = model.predict(imgs_mask_valid_predict, verbose=1)

    imgs_valid_predict *= 2000
    imgs_valid_predict += 1000

    imgs_mask_valid_predict *= 2000
    imgs_mask_valid_predict += 1000

    predictions *= 2000
    predictions += 1000

    # Draw image
    draw_image(predictions, imgs_valid_predict, imgs_mask_valid_predict, displayname)
    return loss



def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
	
# train the network
def train_unet(model, imgs_train, imgs_mask_train):


    # make 4-D array
    imgs_train = imgs_train[..., np.newaxis]
    imgs_mask_train = imgs_mask_train[..., np.newaxis]

    # normalize the image data
    imgs_train = imgs_train.astype('float32')
    #mean_image = np.mean(imgs_train)  # mean for data centering
    #std_image = np.std(imgs_train)  # std for data normalization
    imgs_train -= 1000
    imgs_train /= 2000

    # normalize the image label data
    imgs_mask_train = imgs_mask_train.astype('float32')
    #mean_mask = np.mean(imgs_mask_train)  # mean for data centering
    #std_mask = np.std(imgs_mask_train)  # std for data normalization
    imgs_mask_train -= 1000
    imgs_mask_train /= 2000

    #plt.imshow(imgs_train[0, :, :,0], vmin = 0, vmax = 2000, cmap='gray')
    #plt.show()
    #plt.imshow(imgs_mask_train[0, :, :, 0], vmin = 0, vmax = 2000, cmap='gray')
    #plt.show()

    # Preemptive rule setter for what to do when we fit
    #model_checkpoint = ModelCheckpoint(filepath='checkpoint-{epoch:02d}-{loss:.2f}.hdf5', monitor='loss',save_best_only=True)
    #print("Model CheckPoint:" + str(model_checkpoint.__dict__))
    # fit the model (save the weights as define above)...try to increase the batch size to fit in GPU (look for warning message that CPU is being used instead of GPU)
    # batch size is per image ... 32 would be the whole volume
    lossHistory = model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=50, verbose=0, shuffle=False, validation_split=0.0)

    # Plot to display Loss
    # plot_loss(lossHistory)

    return model
    # Model is also saved after the fit

def fetch_and_preprocess_goodimage(name):

    good_image_gt_key = "CT_datasets/"+name+".nrrd"
    good_image_recon_key = "Recons/recon"+name+".nrrd"
    good_image_truth = name + ".truth.nrrd"
    good_image_train = name + ".train.nrrd"
    

    subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'companydatasets', '--key', good_image_gt_key, good_image_truth])
    subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'companydatasets', '--key', good_image_recon_key,good_image_train])

    good_image_gtVolume, options_good_image_gt = nrrd.read(good_image_truth)
    good_image_trainVolume, options_good_imageTrain = nrrd.read(good_image_train)

    # Transpose the matrices so that the splits attain shape of {32, 256, 256}

    good_image_gtVolume = np.swapaxes(good_image_gtVolume, 0, 2)
    good_image_gtVolume = np.transpose(good_image_gtVolume, (0, 2, 1))

    good_image_trainVolume = np.swapaxes(good_image_trainVolume, 0, 2)
    good_image_trainVolume = np.transpose(good_image_trainVolume, (0, 2, 1))

    # Finding central slices

    length_volume = len(good_image_gtVolume)
    num_of_slices = length_volume / 8
    start = (length_volume / 2) - (num_of_slices / 2)
    end = (length_volume / 2) + (num_of_slices / 2)

    good_image_gtSlices = good_image_gtVolume[start:end, :, :]
    good_image_trainSlices = good_image_trainVolume[start:end, :, :]

    return good_image_gtSlices,good_image_trainSlices

def main():

    finished = False
    studyBucket = "Recons"
    batchSize = 1  # One 3D dataset per batch

    # Initial the model for the first time
    model = get_net()
    #print(model.summary())
    processedBatches = 1	# Counter for monitoring progress

    # A "good" nrrd to test for quality of training
    good_image_name = "1.2.392.200036.9116.2.2.2.1762657837.1489996273.355622"
    good_image_gtSlices, good_image_trainSlices = fetch_and_preprocess_goodimage(good_image_name)


    while (not finished):

        # list all objects in studyBucket
        objectlistCMD = "aws s3api list-objects-v2 --bucket companydatasets --prefix " + studyBucket + " --query '[Contents[].{Key: Key}]'"

        #Find the name of the file
        objectsInStudy = subprocess.Popen(objectlistCMD, shell=True, stdout=subprocess.PIPE)
        std_out, std_error = objectsInStudy.communicate()
        data = json.loads(std_out)
        basekey = data[0][processedBatches]['Key']
        basekey = basekey.replace("Recons/recon", "")
        if(basekey != good_image_name):
            print("Training on File: {}".format(basekey))

            # Constructing recon and GT names
            gt = basekey.replace(".nrrd", ".truth.nrrd")
            train = basekey.replace(".nrrd", ".train.nrrd")
            reconKey = "Recons/recon" + basekey.replace(".nrrd", "") + ".nrrd"
            gtkey = "CT_datasets/" + basekey

            # Fetch the files from aws
            subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'companydatasets', '--key', gtkey, gt])
            subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'companydatasets', '--key', reconKey, train])

            # read the volumes

            gtVolume, optionsGT = nrrd.read(gt)
            trainVolume, optionsTrain = nrrd.read(train)

            # Transpose the matrices so that the splits attain shape of {32, 256, 256}

            gtVolume = np.swapaxes(gtVolume, 0, 2)
            gtVolume = np.transpose(gtVolume, (0, 2, 1))

            trainVolume = np.swapaxes(trainVolume, 0, 2)
            trainVolume = np.transpose(trainVolume, (0, 2, 1))

            # Finding central slices
            if (gtVolume.shape == trainVolume.shape):
                length_volume = len(gtVolume)
                num_of_slices = length_volume / 8
                start = (length_volume / 2) - (num_of_slices / 2)
                end = (length_volume / 2) + (num_of_slices / 2)

                gtSlices = gtVolume[start:end, :, :]
                trainSlices = trainVolume[start:end, :, :]

                # Plots to display slices
                #plt.imshow(gtSlices[0, :, :], vmin = 0, vmax = 2000, cmap='gray')
                #plt.show()
                #plt.imshow(trainSlices[0, :, :], vmin = 0, vmax = 2000, cmap='gray')
                #plt.show()


                if (processedBatches % 10 == 9):

                    # TODO: Run validation on accumulated data
                    # create an NRRD of all images
                    # compute loss
                    # think about how to make this a cumulative process, maybe this is a third process

                    # Run validation on 10% data
                    print('Skipped batch for validation'.format(processedBatches))
                    loss = validate_and_predict(model, gtSlices, trainSlices, "Pred_data"+str(processedBatches))
                    print('Loss at {} batch is: {}'.format(processedBatches * batchSize, loss))

                    # Test model on one good image to assesws quality of training over time
                    loss = validate_and_predict(model, good_image_gtSlices, good_image_trainSlices, "GoodImage_data"+str(processedBatches))
                    print('Loss on Good Image at {} batch is: {}'.format(processedBatches * batchSize, loss))


                else:
                    # train the network
                    model = train_unet(model, gtSlices, trainSlices)
                    os.remove(gt)
                    os.remove(train)

                    # alternative save model from tensorflow https://www.tensorflow.org/programmers_guide/saved_model
                    #  Keras callbacks https://keras.io/callbacks/
                    #  Keras https://keras.io/getting-started/faq/

                #  Monitor the loss from fit function
                print('{} datasets processed, loss at {}, File {}'.format(processedBatches * batchSize, 'TBD',
                                                                          basekey))
            else:
                #print("Blah")
                print(
                    'Shape of train file is {} ; Shape of recon file is {}. Dimensions do not match. Check the files at token {}'.format(
                        gtVolume.shape, trainVolume.shape, basekey))
        else:
            print("Skipping training on Good image")

        if (processedBatches == len(data[0])):
            finished = True
            print("Finished! Your next token is the initial token.")

        processedBatches += 1


    print("Complete.")

    # # Parameters
    # nextToken = 'eyJDb250aW51YXRpb25Ub2tlbiI6IG51bGwsICJib3RvX3RydW5jYXRlX2Ftb3VudCI6IDV9'
    # #nextToken = 'eyJib3RvX3RydW5jYXRlX2Ftb3VudCI6IDUyLCAiQ29udGludWF0aW9uVG9rZW4iOiBudWxsfQ=='
    # batchSize = 1		# One 3D dataset per batch
    # processedBatches = 0	# Counter for monitoring progress
    #
    # # Initial the model for the first time
    # model = get_net()
    # #print(model.summary())
    #
    # if(len(sys.argv) > 1):
    #     nextToken = str(sys.argv[1])
    #
    # # Process batches
    # initialToken = nextToken
    # print ("Current Token:"+str(nextToken))
    # finished = False
    # while(not finished):
    #     # List objects in current batch
    #     listingProc = subprocess.Popen(['aws', 's3api', 'list-objects-v2', '--bucket', 'companydatasets', '--starting-token', nextToken, '--max-items', str(batchSize), '--prefix', 'CT_datasets/', '--query', '[Contents[].{Key: Key}, NextToken]'],stdout=subprocess.PIPE)
    #
    #     # We may need to skip the first folder /
    #     std_out, std_error = listingProc.communicate()
    #     data = json.loads(std_out)
    #     keys = data[0]
    #     nextToken = data[1]
    #
    #     # Download the batch
    #     for k in range(0, len(keys)):
    #         key = keys[k]['Key']
    #         basekey = key.replace("CT_datasets/", "").replace(".nrrd", "")
    #         gt = basekey + ".truth.nrrd"
    #         train = basekey + ".train.nrrd"
    #         reconKey = "Recons/recon" + basekey + ".nrrd"
    #
    #         subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'companydatasets', '--key',  key, gt])
    #         subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'companydatasets', '--key',  reconKey, train])
    #
    #         # read the volumes
    #         gtVolume, optionsGT = nrrd.read(gt)
    #         trainVolume, optionsTrain = nrrd.read(train)




if __name__ == "__main__" :
    main()
		
		
	  



            


	
