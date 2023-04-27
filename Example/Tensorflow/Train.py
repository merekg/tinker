import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import utils
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import Counter
np.random.seed(23)


# utils.createDataset()
# # utils.createTestDataset()

data, target = utils.loadDataset('/media/save//MLVERTEBRAELABELLING/Data/trainData.npy', '/media/save//MLVERTEBRAELABELLING/Data/trainLabels.npy', '/media/save//MLVERTEBRAELABELLING/Data/trainFileName.txt')
# print(data.shape, target.shape)

# model = utils.createModel()
# model.summary()
# model_path = '/media/save//MLVERTEBRAELABELLING/Models/testmodel.h5'
# checkpoints = keras.callbacks.ModelCheckpoint(model_path, monitor = 'loss', verbose = 0, save_best_only = True, save_weights_only = False, mode = 'min')
# callbacks_list = [checkpoints]
# print('Created the model')

# model.fit(x = data, y = target, epochs = 100, batch_size = 16, verbose = 1, shuffle = True, callbacks = callbacks_list, validation_split=0.1)

model = keras.models.load_model('/media/save//MLVERTEBRAELABELLING/Models/testmodel.h5')
utils.predictionsDataset(data, model, target)


print('Finished Training')
print()



# testData, testTarget = utils.loadDataset('/media/save//MLVERTEBRAELABELLING/Data/testData.npy', '/media/save//MLVERTEBRAELABELLING/Data/testLabels.npy', '/media/save//MLVERTEBRAELABELLING/Data/testFileName.txt')
# print(testData.shape, testTarget.shape)


# utils.predictionsDataset(testData, model, testTarget)


# predictions = model.predict(data)
# for i in range(predictions.shape[0]):
# 	print(predictions[i].max(), predictions.min())

# homeDirectory = '/media/save//'
# # homeDirectory = '/home/company/Desktop//'
# homeDirectory = '/home/ubuntu/Desktop/

# predictionDirectory = homeDirectory + 'MLVERTEBRAELABELLING/Predictions/'
# thresholdedPredictionDirectory = homeDirectory + 'MLVERTEBRAELABELLING/ThresholdPredictions/'


# dataDirectory = homeDirectory + 'MLVERTEBRAELABELLING/AugmentedData/'
# labelDirectory = '/home//MLVERTEBRAELABELLING/Labels/NewLabels/'

# testDataDirectory = homeDirectory + 'MLVERTEBRAELABELLING/TestingData/'
# dataset, labels = utils.getDataLabelPair(dataDirectory, labelDirectory)


# print('Gathered the dataset')
# # model = utils.createModel()


# # # /media/save//MLVERTEBRAELABELLING/Models/model_11_Feb.h5
# # /media/save//MLVERTEBRAELABELLING/Models/Tested/model_16_FebFull.h5
# # /media/save//MLVERTEBRAELABELLING/Models/model_full_run.h5
# model = keras.models.load_model(homeDirectory + 'MLVERTEBRAELABELLING/Models/model_16_FebFull.h5')


# for layer in model.layers:
#     print(layer.output_shape)


# # model_path = homeDirectory + 'MLVERTEBRAELABELLING/Models/model_16_Feb.h5'    
# # checkpoints = keras.callbacks.ModelCheckpoint(model_path, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min')
# # callbacks_list = [checkpoints]
# # print('Created the model')

# # model.fit(x = dataset, y = labels, epochs=275, batch_size=2, verbose=1, shuffle=True,callbacks=callbacks_list, validation_split = 0.2)
# print('Finished Training')
# # model.save(homeDirectory + 'MLVERTEBRAELABELLING/Models/model_16_FebFull.h5')


# print('Starting Predictions')
# # utils.predictions(testDataDirectory, model)
# # utils.predictions(dataDirectory, model)
# #utils.predictionsDataset(dataset, model)
# utils.calculateModelMetrics(dataset, labels, model)
