#Import the necessary libraries
from __future__ import absolute_import, division, print_function

#Suppress warnings
import logging
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True

# TensorFlow and tf.keras
import tensorflow as tf
import keras

# Helper libraries
import numpy as np
import h5py
import skimage
from keras.models import load_model
from keras.utils import multi_gpu_model
import time
from keras.backend.tensorflow_backend import set_session

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
	from tensorflow.python.framework.graph_util import convert_variables_to_constants
	graph = session.graph
	with graph.as_default():
		freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
		output_names = output_names or []
		output_names += [v.op.name for v in tf.global_variables()]
		input_graph_def = graph.as_graph_def()
		if clear_devices:
			for node in input_graph_def.node:
				node.device = ""
		#print(input_graph_def)
		print(output_names)
		#print(freeze_var_names)
		frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
	return frozen_graph

model = load_model('/home/ubuntu/Desktop/MachineLearningCTScanInput/-saved-model-3dimage-reconstruction.hdf5')
from keras import backend as K
frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, '/opt/rt3d/ML/', 'imageQuality.pb', as_text=False)
