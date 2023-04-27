import h5py
import numpy as np
import shutil
import os
import tensorflow as tf
from tensorflow import keras
from collections import Counter
from scipy.ndimage import zoom
from tensorflow.keras import backend as K
from skimage.transform import resize
import json
import pymysql
import transforms
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle

HOST = 'localhost'
USER = 'root'
PASSWORD = ''
DATABASE = 'nView'
np.random.seed(23)
# tf.random.set_random_seed(23)

resolution = 64
spacing = 4.6875
threshold = 0.8

homeDirectory = '/media/save/Shalin/'
# homeDirectory = '/home/nview/Desktop/Shalin/'
def getVolume(filePath):
    voxel = np.fromfile(filePath, dtype='uint8')
    voxelData = np.reshape(voxel, (416,416,416))
    return voxelData


def createDataset():
    labelsFilePath = '/media/save/Shalin/MLVERTEBRAELABELLING/Labels/'
    volumePath = '/media/save/Shalin/SmartScrolling/Volumes/'
    X = []
    Y = []
    fileName = []
    for file in os.listdir(labelsFilePath):
        name, ext = file.split('.')
        volume = getVolume(volumePath + name + '.dat')
        resized = resize(volume, (resolution, resolution, resolution), mode = 'constant', preserve_range = True, anti_aliasing=False).astype('uint16')
        volumeResized = resized/255.0
        volumeToAugment = volumeResized
        volumeResized = np.expand_dims(volumeResized, 3)

        label = np.load(labelsFilePath + file);
        target = np.zeros((resolution, resolution, resolution))
        target = target.astype('int')
        for i in range(0, label.shape[0]):
            x = round(float(label[i][0]) / spacing)
            y = round(float(label[i][1]) / spacing)
            z = round(float(label[i][2]) / spacing)
            target[x-1:x+1, y-1:y+1, z-1:z+1] = 1
            # target[x,y,z ] =1
        target = np.transpose(target, (2,1,0))
        targetToAugment = target        
        target = np.expand_dims(target, 3)
        X.append(volumeResized)
        Y.append(target)
        fileName.append(name)
        visualizeDataLabel(volumeResized, target, name)

        theta   = float(np.around(np.random.uniform(-30.0, 30.0, size=1), 2))
        rotatedVol  = transforms.rotateit(volumeToAugment, theta)
        rotatedLabel = transforms.rotateit(targetToAugment, theta, isseg=True)
        name = name + 'r'
        visualizeDataLabel(rotatedVol, rotatedLabel, name)

        rotatedVol = np.expand_dims(rotatedVol, 3)
        rotatedLabel = np.expand_dims(rotatedLabel, 3)

        X.append(rotatedVol)
        Y.append(rotatedLabel)
        fileName.append(name)

        scalefactor  = float(np.around(np.random.uniform(0.7, 1.3, size=1), 2))
        scaledVol  = transforms.scaleit(volumeToAugment, scalefactor)
        scaledLabel = transforms.scaleit(targetToAugment, scalefactor, isseg=True)

        name = name + 's'
        visualizeDataLabel(scaledVol, scaledLabel, name)

        scaledVol = np.expand_dims(scaledVol, 3)
        scaledLabel = np.expand_dims(scaledLabel, 3)

        X.append(scaledVol)
        Y.append(scaledLabel)
        fileName.append(name)

        factor  = float(np.around(np.random.uniform(0.8, 1.2, size=1), 2))
        intensifiedVol  = transforms.intensifyit(volumeToAugment, factor)

        name = name + 'i'
        visualizeDataLabel(intensifiedVol, target, name)

        intensifiedVol = np.expand_dims(intensifiedVol, 3)

        X.append(intensifiedVol)
        Y.append(target)
        fileName.append(name)

        offset  = list(np.random.randint(-10,10, size=3))
        currseg = targetToAugment
        translatedVol  = transforms.translateit(volumeToAugment, offset)
        translatedLabel = transforms.translateit(targetToAugment, offset, isseg=True)

        name = name + 'o'
        visualizeDataLabel(translatedVol, translatedLabel, name)

        translatedVol = np.expand_dims(translatedVol, 3)
        translatedLabel = np.expand_dims(translatedLabel, 3)

        X.append(translatedVol)
        Y.append(translatedLabel)
        fileName.append(name)
    data = np.asarray(X)
    targets = np.asarray(Y)
    fileList = np.asarray(fileName)

    print(data.shape, targets.shape)
    np.save('/media/save/Shalin/MLVERTEBRAELABELLING/Data/trainData.npy', data)
    np.save('/media/save/Shalin/MLVERTEBRAELABELLING/Data/trainLabels.npy', targets)
    np.savetxt('/media/save/Shalin/MLVERTEBRAELABELLING/Data/trainFileName.txt', fileList, fmt="%10s")


def createTestDataset():
    labelsFilePath = '/media/save/Shalin/MLVERTEBRAELABELLING/TestLabels/'
    volumePath = '/media/save/Shalin/SmartScrolling/TestVolumes/'
    X = []
    Y = []
    fileName = []
    for file in os.listdir(labelsFilePath):
        name, ext = file.split('.')
        volume = getVolume(volumePath + name + '.dat')
        resized = resize(volume, (resolution, resolution, resolution), mode = 'constant', preserve_range = True, anti_aliasing=False).astype('uint16')
        volumeResized = resized/255.0
        volumeToAugment = volumeResized
        volumeResized = np.expand_dims(volumeResized, 3)

        label = np.load(labelsFilePath + file);
        target = np.zeros((resolution, resolution, resolution))
        target = target.astype('int')
        for i in range(0, label.shape[0]):
            x = round(float(label[i][0]) / spacing)
            y = round(float(label[i][1]) / spacing)
            z = round(float(label[i][2]) / spacing)
            target[x-2:x+2, y-2:y+2, z-2:z+2] = 1
            # target[x,y,z ] =1
        target = np.transpose(target, (2,1,0))
        targetToAugment = target
        target = np.expand_dims(target, 3)
        X.append(volumeResized)
        Y.append(target)
        fileName.append(name)
    data = np.asarray(X)
    targets = np.asarray(Y)
    fileList = np.asarray(fileName)

    print(data.shape, targets.shape)
    np.save('/media/save/Shalin/MLVERTEBRAELABELLING/Data/testData.npy', data)
    np.save('/media/save/Shalin/MLVERTEBRAELABELLING/Data/testLabels.npy', targets)
    np.savetxt('/media/save/Shalin/MLVERTEBRAELABELLING/Data/testFileName.txt', fileList, fmt="%10s")


def loadDataset(dataFilePath, targetFilePath, fileListPath):
    data = np.load(dataFilePath)
    target = np.load(targetFilePath)
    fileList = np.genfromtxt(fileListPath)
    data, target, fileList = shuffle(data, target, fileList)
    np.savetxt(fileListPath, fileList, fmt="%10s")

    return data, target

def createModel():
    inputs = keras.layers.Input(shape = (resolution,resolution,resolution,1))
    conv1 = keras.layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = keras.layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv1)
    # conv1 = keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv2)
    # conv2 = keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3)
    # conv3 = keras.layers.BatchNormalization()(conv3)
    pool3 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
    # conv4 = keras.layers.BatchNormalization()(conv4)

    up5 = keras.layers.concatenate([keras.layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
    conv5 =  keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up5)
    conv5 =  keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv5)
    # conv5 = keras.layers.BatchNormalization()(conv5)

    up6 = keras.layers.concatenate([keras.layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv2], axis=4)
    conv6 =  keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 =  keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv6)
    # conv6 = keras.layers.BatchNormalization()(conv6)

    up7 = keras.layers.concatenate([keras.layers.Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv1], axis=4)
    conv7 =  keras.layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 =  keras.layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv7)
    # conv7 = keras.layers.BatchNormalization()(conv7)
    conv8 =  keras.layers.Conv3D(1, ( 1, 1, 1), padding='same', activation='sigmoid')(conv7)

    model = keras.models.Model(inputs=inputs, outputs=conv8)
    #model = keras.utils.multi_gpu_model(model, gpus=4)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0005), metrics=['accuracy'], loss='binary_crossentropy')

    return model

def predictionsDataset(dataset, model, target):
    shape = dataset.shape
    for i in range(0, shape[0]):
        data = np.expand_dims(dataset[i], 0)
        d = dataset[i]
        p = model.predict(data)
        print(p.max(), p.min(), p.shape)
        p = np.squeeze(p, 4)
        p = np.squeeze(p, 0)
        p = (p>threshold)
        visualizeDataPred(d, p, (i+1)*100)
        p = suppress(p)
        visualizeDataPred(d, p, i)
        result = np.where(p >= 1)
        mlPredictions = []
        listOfCoordinates= list(zip(result[0]*spacing, result[1]*spacing, result[2]*spacing, p[result[0], result[1], result[2]]))
        for cord in listOfCoordinates:
            jsonDict = {'label': 'V', 'location': [{'0': cord[0], '1': cord[1], '2': cord[2], '3': 1}]}
            mlPredictions.append(jsonDict)
        print(mlPredictions)
        # print(len(listOfCoordinates), 'number of coordinates')
        print('##########################################################################################################')

#     # listOfCoordinates= list(zip(result[0]*2.34375, result[1]*2.34375, result[2]*2.34375, p[result[0], result[1], result[2]]))
#     # for cord in listOfCoordinates:
#     #     print(cord)
#     #     jsonDict = {'label': str(cord[3]), 'location': {'0': cord[0], '1': cord[1], '2': cord[2], '3': 1}}
#     #     mlPredictions.append(jsonDict)

def suppress(p):
    for i in range(0, resolution):
        for j in range(0, resolution):
            for k in range(0, resolution):
                if p[i, j, k] > threshold:
                    temp = p[i, j, k]
                    p[i-5: i+5, j-5:j+5, k-5:k+5 ] = 0
                    p[i,j,k] = temp
    return p

def visualizeDataLabel(data, label,file):
    fig, ax = plt.subplots(2, 2)
    # dataTwo = np.amax(data, axis = 2)
    ax[0][0].imshow(np.amax(data, axis = 0))
    ax[0][1].imshow(np.amax(label, axis = 0))
    ax[1][0].imshow(np.amax(data, axis = 1))
    ax[1][1].imshow(np.amax(label, axis = 1))
    fig.savefig('/media/save/Shalin/MLVERTEBRAELABELLING/DataVisualization/'+str(file)+'.png')
    plt.close(fig) 

def visualizeDataPred(data, label,file):
    fig, ax = plt.subplots(2, 2)
    # dataTwo = np.amax(data, axis = 2)
    ax[0][0].imshow(np.amax(data, axis = 0))
    ax[0][1].imshow(np.amax(label, axis = 0))
    ax[1][0].imshow(np.amax(data, axis = 1))
    ax[1][1].imshow(np.amax(label, axis = 1))
    fig.savefig('/media/save/Shalin/MLVERTEBRAELABELLING/PredVisualisation/'+str(file)+'.png')
    plt.close(fig) 

# def populateDatabase(sourceDirectory):
#     conn = pymysql.connect(host = HOST, user = USER, password = PASSWORD, database = DATABASE)
#     for file in os.listdir(sourceDirectory):
#         if file.endswith('.txt'):
#             reconID =  file.split('.')[0]
#             openedFile = open(sourceDirectory + file,"r+")
#             coordinates = openedFile.read()
#             jsonCoordinates = json.loads(coordinates.strip())
#             cursor = conn.cursor()
#             print(reconID)
#             query = "update reconstructions set vertebraePredicted = \'" + coordinates + "\' where ReconID = " + reconID + ";"
#             cursor.execute(query)
#             conn.commit()
            
            
    


# def testAugmentations():
    
#     sourceFile = '/home/tristanmary/Desktop/Shalin/MLVERTEBRAELABELLING/Test/testLabel.h5'
#     dataFile = '/home/tristanmary/Desktop/Shalin/MLVERTEBRAELABELLING/Test/2038993828Label.h5'
#     labelFile = '/home/tristanmary/Desktop/Shalin/MLVERTEBRAELABELLING/AugmentedLabels/2038993828.h5'

#     augmentedFileDataOpen = h5py.File(dataFile, 'r+')
#     augmentedFileLabelOpen = h5py.File(sourceFile, 'r+')
#     dataVoxel = augmentedFileDataOpen['ITKImage/0/VoxelData']
#     labelVoxel = augmentedFileLabelOpen['ITKImage/0/VoxelData']
#     dataVoxelProjections = dataVoxel[...]
#     labelVoxel[...] = dataVoxelProjections
    
# def getMax(directory):
#     globalMaximum = 0
#     for file in os.listdir(directory):
#         openedFile = h5py.File(directory + file, 'r+')
#         dataVoxel = openedFile['ITKImage/0/VoxelData']
#         dataVoxelProjections = dataVoxel[...]
#         currentMaximum = dataVoxelProjections.max()
#         if currentMaximum > globalMaximum:
#             globalMaximum = currentMaximum
        
#     return globalMaximum
        
# def getMin(directory):
#     globalMinimium = 500
#     for file in os.listdir(directory):
#         openedFile = h5py.File(directory + file, 'r+')
#         dataVoxel = openedFile['ITKImage/0/VoxelData']
#         dataVoxelProjections = dataVoxel[...]
#         currentMinimum = dataVoxelProjections.min()
#         if currentMinimum < globalMinimium:
#             globalMinimium = currentMinimum
        
#     return globalMinimium

# def getDataLabelPair(dataDirectory, labelDirectory):
#     XTrainList = []
#     YTrainList = []
#     for file in os.listdir(dataDirectory):
#         print('Addressing file', file)
#         augmentedFileDataOpen = h5py.File(dataDirectory + file, 'r+')
#         dataVoxel = augmentedFileDataOpen['ITKImage/0/VoxelData']
#         dataVoxelProjections = dataVoxel[...]
#         data = resize(dataVoxelProjections, (128, 128, 128), mode = 'constant', preserve_range = True)
#         data = data/255

#         labels = np.zeros((128,128,128))
#         labels = np.transpose(labels, (2,1,0))
#         voxelDataAttrs = augmentedFileDataOpen['ITKImage/0/VoxelData'].attrs
#         vertebraeIdentifiedDictionary = eval(voxelDataAttrs['vertebraeIdentified'])
#         for i in range(0, len(vertebraeIdentifiedDictionary)):
#             locationDictionary = vertebraeIdentifiedDictionary[i]['location']
#             label = vertebraeIdentifiedDictionary[i]['label']
#             x = round(float(locationDictionary['0']) / 2.34375)
#             y = round(float(locationDictionary['1']) / 2.34375)
#             z = round(float(locationDictionary['2']) / 2.34375)
#             if(label.startswith('T')):
#                 #labels[x-1: x+1, y-1:y+1, z-1: z+1] = 1
#                 labels[x,y,z] = 1
#             elif (label.startswith('L')):
#                 #labels[x-1: x+1, y-1:y+1, z-1: z+1 ] = 1
#                 labels[x,y,z] = 1

#         toAugmentData = data
#         toAugmentLabel = labels

#         # visualizeDataLabel(data, labels, file)

#         data = np.expand_dims(data, 3)
#         label = np.expand_dims(labels, 3)

#         print(data.max(), data.min(), label.max(), label.min())
#         XTrainList.append(data)
#         YTrainList.append(label)
        
#         for i in range(0, 10):
#             theta   = float(np.around(np.random.uniform(-90.0,90.0, size=1), 2))
#             dataRotated = transforms.rotateit(toAugmentData, theta)
#             dataRotated[dataRotated < 0] = 0
#             labelRotated = transforms.rotateit(toAugmentLabel, theta, isseg=True)
#             print(dataRotated.max(), dataRotated.min(), labelRotated.max(), labelRotated.min())
#             # visualizeDataLabel(dataRotated, labelRotated, 'no')

#             dataRotated = np.expand_dims(dataRotated, 3)
#             labelRotated = np.expand_dims(labelRotated, 3)
#             XTrainList.append(dataRotated)
#             YTrainList.append(labelRotated)

#             scalefactor  = float(np.around(np.random.uniform(0.75, 1.25, size=1), 2))
#             dataScaled  = transforms.scaleit(toAugmentData, scalefactor)
#             dataScaled[dataScaled < 0] = 0
#             labelScaled = transforms.scaleit(toAugmentLabel, scalefactor, isseg=True)
#             # visualizeDataLabel(dataScaled, labelScaled, 'no')
#             dataScaled = np.expand_dims(dataScaled, 3)
#             labelScaled = np.expand_dims(labelScaled, 3)
#             print(dataScaled.max(), dataScaled.min(), labelScaled.max(), labelScaled.min())
#             XTrainList.append(dataScaled)
#             YTrainList.append(labelScaled)


#             factor  = float(np.around(np.random.uniform(0.8, 1.4, size=1), 2))
#             dataIntensified  = transforms.intensifyit(toAugmentData, factor)
#             dataIntensified[dataIntensified < 0] = 0
#             labelIntensified = toAugmentLabel
#             # visualizeDataLabel(dataIntensified, labelIntensified, 'no')
#             dataIntensified = np.expand_dims(dataIntensified, 3)
#             labelIntensified = np.expand_dims(labelIntensified, 3)

#             print(dataIntensified.max(), dataIntensified.min(), labelIntensified.max(), labelIntensified.min())
#             XTrainList.append(dataIntensified)
#             YTrainList.append(labelIntensified)

#     Xtrain = np.asarray(XTrainList)
#     Ytrain = np.asarray(YTrainList)
#     print('Created arrays')

#     return Xtrain, Ytrain

   

        
# def createModel():
#     inputs = keras.layers.Input(shape = (128,128,128,1))
#     conv1 = keras.layers.Conv3D(32, (3, 3, 3), activation='relu',padding='same')(inputs)
#     conv1 = keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
#     # conv1 = keras.layers.BatchNormalization()(conv1)
#     pool1 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

#     conv2 = keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
#     conv2 = keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
#     # conv2 = keras.layers.BatchNormalization()(conv2)
#     pool2 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

#     conv3 = keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
#     conv3 = keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
#     # conv3 = keras.layers.BatchNormalization()(conv3)
#     pool3 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

#     conv4 = keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
#     conv4 = keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
#     # conv4 = keras.layers.BatchNormalization()(conv4)


#     up5 = keras.layers.concatenate([keras.layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
#     conv5 =  keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up5)
#     conv5 =  keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv5)
#     # conv5 = keras.layers.BatchNormalization()(conv5)

#     up6 = keras.layers.concatenate([keras.layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv2], axis=4)
#     conv6 =  keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up6)
#     conv6 =  keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv6)
#     # conv6 = keras.layers.BatchNormalization()(conv6)

#     up7 = keras.layers.concatenate([keras.layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv1], axis=4)
#     conv7 =  keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up7)
#     conv7 =  keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv7)
#     # conv7 = keras.layers.BatchNormalization()(conv7)
#     conv8 =  keras.layers.Conv3D(1, ( 1, 1, 1), padding='same', activation='sigmoid')(conv7)

#     model = keras.models.Model(inputs=inputs, outputs=conv8)
#     #model = keras.utils.multi_gpu_model(model, gpus=4)
#     model.compile(optimizer=keras.optimizers.Adam(lr=0.0005), metrics=['accuracy'], loss='binary_crossentropy')

#     return model




# def calculateModelMetrics(dataset, labels, model):
#     shape = dataset.shape
#     totalError = 0
#     totalMissed = 0
#     totalExtra = 0
#     for i in range(0, shape[0]):
#         data = np.expand_dims(dataset[i], 0)
#         p = model.predict(data)
#         p = np.squeeze(p, 4)
#         p = np.squeeze(p, 0)
#         p = suppressThreshold(p, 0.8)
#         l = labels[i]
#         l = np.squeeze(l, 3)
#         #l = suppressThreshold(l, 0.9)
#         resultPredicted = np.where(p > 0.8)
#         listOfCoordinatesPredicted = list(zip(resultPredicted[0]*2.34375, resultPredicted[1]*2.34375, resultPredicted[2]*2.34375, p[resultPredicted[0], resultPredicted[1], resultPredicted[2]]))
#         resultLabel = np.where(l == 1)
#         listOfCoordinatesLabel = list(zip(resultLabel[0]*2.34375, resultLabel[1]*2.34375, resultLabel[2]*2.34375, p[resultLabel[0], resultLabel[1], resultLabel[2]]))
#         extraVertebrae, missedVertebrae, avgError =  calculateMetrics(listOfCoordinatesPredicted, listOfCoordinatesLabel)
#         totalError += avgError
#         totalMissed += missedVertebrae
#         totalExtra += extraVertebrae
#     print ('Average Error: ',totalError/shape[0], ' Total Vertebrae missed: ', totalMissed, ' Total Vertebrae Extra: ', totalExtra )
        


# def calculateMetrics(listOfCoordinatesPredicted, listOfCoordinatesLabel):
#     labelLength = len(listOfCoordinatesLabel)
#     predictedLength = len(listOfCoordinatesPredicted)
#     print(labelLength, predictedLength)
#     totalError = 0
#     extraVertebrae = 0
#     missedVertebrae = 0
#     if labelLength < predictedLength:
#         extraVertebrae = predictedLength - labelLength
#         r = labelLength
#     elif labelLength > predictedLength:
#         missedVertebrae = labelLength - predictedLength
#         r = predictedLength
#     else:
#         r = predictedLength
#     print(r)
    
#     for i in range(0,r):
#         xdiff = listOfCoordinatesPredicted[i][0] - listOfCoordinatesLabel[i][0]
#         ydiff = listOfCoordinatesPredicted[i][1] - listOfCoordinatesLabel[i][1]
#         zdiff = listOfCoordinatesPredicted[i][2] - listOfCoordinatesLabel[i][2]
#         xdiff = pow(xdiff, 2)
#         ydiff = pow(ydiff, 2)
#         zdiff = pow(zdiff, 2)
#         distancePerPoint = math.sqrt(xdiff + ydiff + zdiff)
#         totalError = totalError + distancePerPoint
#     if r != 0:
#         avgError = totalError/r
#     else:
#         avgError = 0
#     return extraVertebrae, missedVertebrae, avgError
    
    
    
    

# def predictions(dataDirectory, model):
#     destinationFolder = homeDirectory + 'MLVERTEBRAELABELLING/TestingPredictions/'
#     for file in os.listdir(dataDirectory):
#         print('Predicting file ', file)
#         augmentedFileDataOpen = h5py.File(dataDirectory + file, 'r+')
#         dataVoxel = augmentedFileDataOpen['ITKImage/0/VoxelData']
#         dataVoxelProjections = dataVoxel[...]
        
               
#         dataVoxelProjections = resize(dataVoxelProjections, (128, 128, 128), mode = 'constant', preserve_range = True)
#         dataVoxelProjections = dataVoxelProjections/ 255         
#         dataVoxelProjections = np.expand_dims(dataVoxelProjections, 3)
#         dataVoxelProjections = np.expand_dims(dataVoxelProjections, 0)
#         print(dataVoxelProjections.max())
#         predictions = model.predict(dataVoxelProjections)
#         processPredictions(predictions, file, destinationFolder, dataVoxelProjections)
#         print('Finished Predicting')
#         print('##########################################################################################################')
        
# def processPredictions(predictions, file, destinationFolder, dataVoxelProjections):
#     p = predictions[0]
#     p = np.squeeze(p, 3)
#     print(p.min(), p.max(), p.shape)
#     data = dataVoxelProjections[0]
#     data = np.squeeze(data, 3)
#     visualizeDataLabel(data,  p, file)

#     # p = suppress(p)
#     # result = np.where(p > 0.5)
#     # mlPredictions = []
#     # listOfCoordinates= list(zip(result[0]*2.34375, result[1]*2.34375, result[2]*2.34375, p[result[0], result[1], result[2]]))
#     # for cord in listOfCoordinates:
#     #     print(cord)
#     #     jsonDict = {'label': str(cord[3]), 'location': {'0': cord[0], '1': cord[1], '2': cord[2], '3': 1}}
#     #     mlPredictions.append(jsonDict)
#     # fileName = file.split('.')[0]
#     # print(destinationFolder+fileName)
#     # with open (destinationFolder + fileName + '.txt', 'w') as outFile:
#     #     json.dump(mlPredictions, outFile)

# def suppressThreshold(p, t):
#     for i in range(0, 128):
#         for j in range(0, 128):
#             for k in range(0, 128):
#                 if p[i, j, k] >= t:
#                     temp = p[i, j, k]
#                     p[i-1: i+1, j-1:j+1, k-1:k+1 ] = 0
#                     p[i,j,k] = temp
#     return p




# def fullsize(dataDirectory, destinationFolder):
#     sourceFile = '/home/shalin/MLVERTEBRAELABELLING/Test/test.h5'
#     for file in os.listdir(dataDirectory):
#         print('Predicting file ', file)
#         augmentedFileDataOpen = h5py.File(dataDirectory + file, 'r+')
#         dataVoxel = augmentedFileDataOpen['ITKImage/0/VoxelData']
#         dataVoxelProjections = dataVoxel[...]
#         shutil.copyfile(sourceFile, destinationFolder + file )
#         openedFile = h5py.File(destinationFolder + file, 'r+')
#         data = openedFile['ITKImage/0/VoxelData']
#         data[...] = dataVoxelProjections


# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)


# def threshold(predictionDirectory, thresholdedPredictionDirectory):
#     for file in os.listdir(predictionDirectory):
#         print('Plotting for ', file)
#         augmentedFileDataOpen = h5py.File(predictionDirectory + file, 'r+')
#         dataVoxel = augmentedFileDataOpen['ITKImage/0/VoxelData']
#         dataVoxelProjections = dataVoxel[...]
#         dataVoxelProjections[dataVoxelProjections < 0.95] = 0.0
#         shutil.copyfile(predictionDirectory + file, thresholdedPredictionDirectory + file)
#         thresholdFileOpen = h5py.File(thresholdedPredictionDirectory + file, 'r+')
#         tDataVoxel = thresholdFileOpen['ITKImage/0/VoxelData']
#         tDataVoxel[...] = dataVoxelProjections
