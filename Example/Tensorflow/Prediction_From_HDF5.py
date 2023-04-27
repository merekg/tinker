import tensorflow as tf
import keras
import DataLoader
from keras.models import load_model
from keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)
dataset = DataLoader.read_data('/home/ubuntu/Desktop/MachineLearning/ViewerMachineLearning/Testing/')
model = load_model('/home/ubuntu/Desktop/MachineLearning/ViewerMachineLearning/Model.hdf5')
result = model.predict(dataset)
print(result)


