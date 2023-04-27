import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import keras
import h5py
from keras.models import load_model
from keras.models import model_from_json
from keras.initializers import glorot_uniform
from random import shuffle
from keras.utils import multi_gpu_model
import shutil
from keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)
gt_dir = '/home/ml/Desktop/Machine_Learning/Ground_Truth/Ground_Truth_256/'

recon_dir = '/home/ml/Desktop/Machine_Learning/Early_Reconstruction/Early_Reconstruction_256/'

volumes = []
labels = []


for i in range(1, 501):
  if i <= 9:
    gt_input = h5py.File(gt_dir + 'GTVol_000' + str(i) + '.h5', 'r')
    recon_input = h5py.File(recon_dir + 'recon' + str(i) + '.h5', 'r')
  if i > 9 and i<=99:
    gt_input = h5py.File(gt_dir + 'GTVol_00'+str(i)+'.h5', 'r')
    recon_input = h5py.File(recon_dir + 'recon' + str(i) + '.h5', 'r')

  if i > 99 and i<= 999:
    gt_input = h5py.File(gt_dir + 'GTVol_0'+str(i)+'.h5', 'r')
    recon_input = h5py.File(recon_dir + 'recon' + str(i) + '.h5', 'r')

  if i > 999:
    gt_input = h5py.File(gt_dir + 'GTVol_'+str(i)+'.h5', 'r')
    recon_input = h5py.File(recon_dir + 'recon' + str(i) + '.h5', 'r')



  
  recon_projections = recon_input['ITKImage/0/VoxelData'][:]
  gt_projections = gt_input['ITKImage/0/VoxelData'][:]



  recon_projections = recon_projections/10922.5 
  gt_projections = gt_projections/10922.5


  
  recon_projections = np.expand_dims(recon_projections, axis= 3)
  gt_projections = np.expand_dims(gt_projections, axis= 3 ) 


  volumes.append(recon_projections)
  labels.append(gt_projections)
  

dataset = np.asarray(volumes)
targets = np.asarray(labels)
print(dataset.shape)
print(targets.shape)




#########################################################################################################
#PLEASE ADD BATCH NORMALIZATION TO THE NETWORK IF WE MOVE TO ACTUAL SPINE DATA
# inputs = keras.layers.Input(shape = (32,32,32,1))

def create_model(learn_rate=0.0005, init_mode='glorot_uniform'):

	inputs = keras.layers.Input(shape = (64,64,64,1))
	conv1 = keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer=init_mode)(inputs)
	conv1 = keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer=init_mode)(conv1)
	# conv1 = keras.layers.BatchNormalization()(conv1)
	pool1 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

	conv2 = keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer=init_mode)(pool1)
	conv2 = keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer=init_mode)(conv2)
	# conv2 = keras.layers.BatchNormalization()(conv2)
	pool2 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

	conv3 = keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer=init_mode)(pool2)
	conv3 = keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer=init_mode)(conv3)
	# conv3 = keras.layers.BatchNormalization()(conv3)
	pool3 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

	conv4 = keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer=init_mode)(pool3)
	conv4 = keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer=init_mode)(conv4)
	# conv4 = keras.layers.BatchNormalization()(conv4)


	up5 = keras.layers.concatenate([keras.layers.Conv3DTranspose(512, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
	conv5 =  keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer=init_mode)(up5)
	conv5 =  keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer=init_mode)(conv5)
	# conv5 = keras.layers.BatchNormalization()(conv5)

	up6 = keras.layers.concatenate([keras.layers.Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv2], axis=4)
	conv6 =  keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer=init_mode)(up6)
	conv6 =  keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer=init_mode)(conv6)
	# conv6 = keras.layers.BatchNormalization()(conv6)

	up7 = keras.layers.concatenate([keras.layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv1], axis=4)
	conv7 =  keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer=init_mode)(up7)
	conv7 =  keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer=init_mode)(conv7)
	# conv7 = keras.layers.BatchNormalization()(conv7)
	up8 = keras.layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
	up9 = keras.layers.Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(up8)


	conv8 =  keras.layers.Conv3D(1, (1, 1, 1), padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer=init_mode, activation='relu')(up9)

	model = keras.models.Model(inputs=inputs, outputs=conv8)

	# model = multi_gpu_model(model, gpus=8)
	model.compile(optimizer=keras.optimizers.Adam(lr=learn_rate), metrics=['accuracy'], loss=keras.losses.mean_squared_error)

	return model


model = KerasClassifier(build_fn=create_model, verbose=0, epochs= 15, batch_size = 10)

print('Model Compiled')

# batch_size = [5, 10, 15]
# epochs = [15, 25, 37, 50]
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
learn_rate = [0.0001, 0.0005, 0.001, 0.005]
# momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(learn_rate=learn_rate, init_mode=init_mode)
# , optimizer=optimizer, learn_rate=learn_rate, init_mode=init_mode,activation=activation 
print('Began Searching for the best parameters')
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(dataset, targets)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))




