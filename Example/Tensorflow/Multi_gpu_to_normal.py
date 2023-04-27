#!/usr/bin/python3
import tensorflow as tf
import os
import keras
from keras.models import load_model
import sys
import Convert_Model_to_pb


def convert_model_to_single(multi_gpu_model, single_gpu_model):
	multi_gpus_model = load_model(multi_gpu_model)
	origin_model = multi_gpus_model.layers[-2]
	Convert_Model_to_pb.pb(origin_model, single_gpu_model) 
	

if __name__ == '__main__':
	convert_model_to_single(multi_gpu_model, single_gpu_model)