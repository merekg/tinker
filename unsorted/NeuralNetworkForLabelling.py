import tensorflow as tf
import keras
import DataLoader
import Model
from keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)

dataset = DataLoader.read_data('/home/ubuntu/Desktop/MachineLearning/ViewerMachineLearning/Augmented/')
print(dataset.shape)
targets = DataLoader.read_labels('/home/ubuntu/Desktop/MachineLearning/ViewerMachineLearning/Augmented/')
print('###############')
print(targets.shape)
print('Building the Model')
model_path = '/home/ubuntu/Desktop/MachineLearning/ViewerMachineLearning/Model.hdf5'
model = Model.create_model()
print('Done Creating Model')
checkpoints = keras.callbacks.ModelCheckpoint(model_path, monitor='accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='min')
callbacks_list = [checkpoints]
model.fit(dataset, targets, batch_size=1, epochs=25, verbose=1,shuffle=True,callbacks=callbacks_list)
model.save('/home/ubuntu/Desktop/MachineLearning/ViewerMachineLearning/Model.hdf5')

