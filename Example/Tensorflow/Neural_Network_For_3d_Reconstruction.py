#!/usr/bin/python3
import nrrd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import keras
import numpy as np
import h5py
from keras.models import load_model

from keras.utils import multi_gpu_model
import DataLoader
import Model
import Multi_gpu_to_normal



import sys, getopt
from keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)

def train(gt_dir, recon_dir, model_path):
  dataset = DataLoader.read_data(gt_dir)
  targets = DataLoader.read_data(recon_dir)

  print('Build the Model')
  model = Model.create_model()


  checkpoints = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min')
  callbacks_list = [checkpoints]
  history = model.fit(dataset, targets, batch_size=10, epochs=50, verbose=1,shuffle=True,callbacks=callbacks_list, validation_split =0.1)


def main(argv):
  gt_dir = ''
  recon_dir = ''
  model_path = ''
  pb_file_dir = ''
  try:
    opts, args = getopt.getopt(argv,"hg:r:m:p:",["gt_dir=","recon_dir=", "model_path=", "pb_file_dir="])
  except getopt.GetoptError:
    print ('Neural_Network_For_3d_Reconstruction.py -g <gt_dir> -r <recon_dir> -m <model_path> -p <pb_file_dir>')
    sys.exit(2)
  for opt, arg in opts:
    if opt in ('-h', '--help'):
      print ('Neural_Network_For_3d_Reconstruction.py -g <gt_dir> -r <recon_dir> -m <model_path> -p <pb_file_dir>')
      sys.exit()
    elif opt in ("-g", "--gt_dir"):
      gt_dir = arg
    elif opt in ("-r", "--recon_dir"):
      recon_dir = arg
    elif opt in ("-m", "--model_path"):
      model_path = arg
    elif opt in ("-p", "--pb_file_dir"):
      pb_file_dir = arg
  train(gt_dir, recon_dir, model_path)
  Multi_gpu_to_normal.convert_model_to_single(model_path, pb_file_dir)


if __name__ == "__main__":
   main(sys.argv[1:])

