import tensorflow as tf
import keras
# from keras.initializers import glorot_uniform
from keras.utils import multi_gpu_model
from keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)

def create_model():
	inputs = keras.layers.Input(shape = (64,64,64,1))
	conv1 = keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(inputs)
	conv1 = keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(conv1)
	# conv1 = keras.layers.BatchNormalization()(conv1)
	pool1 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

	conv2 = keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(pool1)
	conv2 = keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(conv2)
	# conv2 = keras.layers.BatchNormalization()(conv2)
	pool2 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

	conv3 = keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(pool2)
	conv3 = keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(conv3)
	# conv3 = keras.layers.BatchNormalization()(conv3)
	pool3 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

	conv4 = keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(pool3)
	conv4 = keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(conv4)
	# conv4 = keras.layers.BatchNormalization()(conv4)


	up5 = keras.layers.concatenate([keras.layers.Conv3DTranspose(512, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
	conv5 =  keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(up5)
	conv5 =  keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(conv5)
	# conv5 = keras.layers.BatchNormalization()(conv5)

	up6 = keras.layers.concatenate([keras.layers.Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv2], axis=4)
	conv6 =  keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(up6)
	conv6 =  keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(conv6)
	# conv6 = keras.layers.BatchNormalization()(conv6)

	up7 = keras.layers.concatenate([keras.layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv1], axis=4)
	conv7 =  keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(up7)
	conv7 =  keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(conv7)
	# conv7 = keras.layers.BatchNormalization()(conv7)
	up8 = keras.layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
	up9 = keras.layers.Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(up8)


	conv8 =  keras.layers.Conv3D(1, (1, 1, 1), padding='same',use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform', activation='relu')(up9)

	model = keras.models.Model(inputs=inputs, outputs=conv8)

	model = multi_gpu_model(model, gpus=8)
	model.compile(optimizer=keras.optimizers.Adam(lr=0.0005), metrics=['accuracy'], loss=keras.losses.mean_squared_error)

	return model

if __name__ == '__main__':
	model = create_model()
