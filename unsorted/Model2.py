import tensorflow as tf
import keras
# from keras.initializers import glorot_uniform
from keras.utils import multi_gpu_model
from keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)

def create_model():
    inputs = keras.layers.Input(shape = (416,416,416,1))
    
    conv1 = keras.layers.Conv3D(filters = 8, kernel_size=(3,3,3), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(inputs)
    pool1 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    conv2 = keras.layers.Conv3D(filters = 16, kernel_size=(3,3,3), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(pool1)
    pool2 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = keras.layers.Conv3D(filters = 32, kernel_size=(3,3,3), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(pool2)
    pool3 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    #conv4 = keras.layers.Conv3D(filters = 64, kernel_size=(3,3,3), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(pool3)
    #pool4 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    
    flat = keras.layers.Flatten()(pool3)
    output = keras.layers.Dense(1, activation = 'sigmoid')(flat)
    
    model = keras.models.Model(inputs=inputs, outputs=output)
    #model = multi_gpu_model(model, gpus=8)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'], loss=keras.losses.binary_crossentropy)
    return model

if __name__ == '__main__':
	model = create_model()
    
