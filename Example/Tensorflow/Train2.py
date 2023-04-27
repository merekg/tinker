import sys, getopt, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, Flatten, Dropout, MaxPool3D
import math
from sklearn.utils import shuffle
# import os
import random
random.seed(23)



np.set_printoptions(suppress=True)


DAT_FILE_PATH = '/home/company/Desktop//Data/'



def getVolume(filePath):
	voxel = np.fromfile(filePath, dtype='uint8')
	voxelData = np.reshape(voxel, (416,416,416))
	return voxelData

def createDataset():
	dataset = []
	targetsList = []
	for i in range(0, 52):
		reconID = labels_recon[i][4]
		filePath = DAT_FILE_PATH + str(int(reconID)) + '.dat'
		volume = getVolume(filePath)
		resized = resize(volume, (64, 64, 64), mode = 'constant', preserve_range = True, anti_aliasing=False).astype('uint16')
		volumeResized = np.expand_dims(resized, 3)
		volumeResized = volumeResized/255.0
		print(volumeResized.shape, np.amax(volumeResized), np.amin(volumeResized))
		parameter = np.delete(labels_recon[i], 4)
		dataset.append(volumeResized)
		targetsList.append(parameter)

	data = np.asarray(dataset)
	target = np.asarray(targetsList)

	return data, target

def loadDataset(dataFilePath, targetFilePath):
	data = np.load(dataFilePath)
	labels_recon = np.genfromtxt(targetFilePath, delimiter = ',')
	print('Finished loading')
	targetsList = []
	for i in range(0, data.shape[0]):
		parameter = np.delete(labels_recon[i], 8)
		# print(parameter.shape)
		parameter[0] = parameter[0] * 1000000
		parameter[1] = parameter[1] * 10000
		parameter[2] = parameter[2] * 10
		parameter[3] = parameter[3] / 10

		parameter[4] = parameter[4] * 1000000
		parameter[5] = parameter[5] * 10000
		parameter[6] = parameter[6] * 10
		parameter[7] = parameter[7] / 10
		targetsList.append(parameter)
	target = np.asarray(targetsList)
	return data, target


def createModel():
	model = Sequential()
	model.add(Conv3D(16, (3, 3, 3), activation='relu',padding='same', input_shape = (64,64,64,1)))
	model.add(MaxPool3D(pool_size = (2,2,2)))
	# model.add(Conv3D(16, (3, 3, 3), activation='relu',padding='same', input_shape = (64,64,64,1)))
	# model.add(MaxPool3D(pool_size = (2,2,2)))
	# model.add(Conv3D(32, (3, 3, 3), activation='relu',padding='same', input_shape = (64,64,64,1)))
	# model.add(MaxPool3D(pool_size = (2,2,2)))
	# model.add(Dropout(0.3))
	model.add(Flatten())
	# model.add(Dense(128))
	model.add(Dense(16))
	# model.add(Dense(32))
	model.add(Dense(8))

	model.compile(optimizer=keras.optimizers.Adam(lr=0.005), metrics=['accuracy'], loss='mean_squared_error')
	return model

def train(data, target, epochs):
	epochs = epochs
	model = createModel()
	model.summary()
	model_path = '/media/save//SmartScrolling/Models/model_cubic' + str(epochs)+'.h5'
	checkpoints = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min')
	callbacks_list = [checkpoints]
	print('Created the model')

	model.fit(x = data, y = target, epochs=epochs, batch_size=16, verbose=1, shuffle=True,callbacks=callbacks_list, validation_split = 0.1)
	print('Finished Training')
	model = keras.models.load_model('/media/save//SmartScrolling/Models/model_cubic' + str(epochs)+'.h5', compile = False)
	predictions = model.predict(data)
	return predictions

def test(data):
	model = keras.models.load_model('/media/save//SmartScrolling/Models/model_cubic.h5', compile = False)
	predictions = model.predict(data)
	return predictions

def calculatePoint(target, x):
	z = ((target[0]/100000) * (x**2)) + ((target[1]/1000) * x) + target[2]
	y = ((target[3]/100000) * (x**2)) + ((target[4]/1000) * x) + target[5]
	return [x,y,z]

def calculateCubicPoint(target, x):
	z = ((target[0]/1000000) * (x**3)) + ((target[1]/10000) * (x**2)) + (target[2]/10 * x) + (target[3] * 10)
	y = ((target[4]/1000000) * (x**3)) + ((target[5]/10000) * (x**2)) + (target[6]/10 * x) + (target[7] * 10)
	return [x,y,z]

def distanceMetric(target, predictions, epochs, test):
	xPoints = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275]
	metric = []
	for i in range(0, target.shape[0]):
		m = []
		total_d = 0
		for x in xPoints:
			t = calculateCubicPoint(target[i], x)
			p = calculateCubicPoint(predictions[i], x)

			d = distanceFormula(t, p)
			# m.append(d)
			total_d += d
		m.append(total_d/len(xPoints))
		metric.append(m)
	metricNp = np.asarray(metric)
	if test:
		np.savetxt("test_metric_cubic" + str(epochs) + ".csv", metricNp, delimiter=",", fmt='%f')
	else:
		np.savetxt("metric_cubic" + str(epochs) + ".csv", metricNp, delimiter=",", fmt='%f')



def getPredictions():
	data = np.load('/media/save//SmartScrolling/Data/CubicDataToTest.npy')
	model = keras.models.load_model('/media/save//SmartScrolling/BestModels/model_cubic250.h5')
	predictions = model.predict(data)
	predictions[:, 0] /= 1000000
	predictions[:, 1] /= 10000
	predictions[:, 2] /= 10
	predictions[:, 3] *= 10
	predictions[:, 4] /= 1000000
	predictions[:, 5] /= 10000
	predictions[:, 6] /= 10
	predictions[:, 7] *= 10
	print(predictions.shape)
	np.savetxt("predictions_full.csv", predictions, delimiter=",", fmt='%f')	



def distanceFormula(p1, p2):
	d = math.sqrt(((p1[0] - p2[0]) **2) +  ((p1[1] - p2[1]) **2) + ((p1[2] - p2[2]) **2))
	return d

# data, target = createDataset()
# np.save('/home/company/Desktop//data.npy', data)
epochs = [150, 200, 250, 300]

data, target = loadDataset('/media/save//SmartScrolling/Data/augmentedCubicTrainData.npy', '/media/save//SmartScrolling/CubicFiles/augmentedTrainLabelsCubic.csv' )
# np.savetxt("targets.csv", target, delimiter=",", fmt='%f')
augmentedTracker = np.genfromtxt('/media/save//SmartScrolling/CubicFiles/augmentedTracker.csv', delimiter = ',')
data, target, augmentedTracker = shuffle(data, target, augmentedTracker)
print(data.shape, target.shape)
np.savetxt("/media/save//SmartScrolling/CubicFiles/augmentedTrackerShuffled.csv", augmentedTracker, delimiter=",", fmt='%s')

# testData, testTarget = loadDataset('/media/save//SmartScrolling/Data/cubicTestData.npy', '/media/save//SmartScrolling/Files/testLabelsCubic.csv' )
# print(testData.shape, testTarget.shape)


# predictions = train(data, target , e)
# distanceMetric(target, predictions, e)
# np.savetxt("Labels_test.csv", target, delimiter=",", fmt='%f')


# for e in epochs:
# 	predictions = train(data, target , e)
# 	distanceMetric(target, predictions, e, False)
	# testPredictions = test(testData)
	# distanceMetric(testTarget, testPredictions, e, True)

getPredictions()



# 0.000621,-0.161106,130.571274,-0.002304,0.692231,151.043381
