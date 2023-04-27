import tensorflow as tf
from tensorflow.python.platform import gfile
import sys
import h5py
import numpy as np
import shutil
import time


f = gfile.FastGFile(str(sys.argv[1]), 'rb')
graph_def = tf.GraphDef()
# Parses a serialized binary message into the current message.
graph_def.ParseFromString(f.read())
f.close()

sess = tf.compat.v1.Session()

sess.graph.as_default()
# Import a serialized TensorFlow `GraphDef` protocol buffer
# and place into the current default `Graph`.
tf.import_graph_def(graph_def)
volumes = []
predicted_projections = []


recon_input = h5py.File(str(sys.argv[2]), 'r')
recon_projections = recon_input['ITKImage/0/VoxelData'][:]
#recon_projections = recon_projections/10922.5

print(recon_projections.min(), recon_projections.max())
recon_projections = np.expand_dims(recon_projections, axis= 3)
volumes.append(recon_projections)
dataset = np.asarray(volumes)
print(dataset.shape)

t0 = time.time()
relu_tensor = sess.graph.get_tensor_by_name('import/dense_1/Sigmoid:0')
predictions = sess.run(relu_tensor, {'import/input_1:0': dataset})
t1 = time.time()
total = t1-t0
print('Time taken to run a single prediction using the .pb file is ', total)

print(predictions)
#shutil.copyfile('/home/ml/Desktop/Machine_Learning/Ground_Truth/GTVol_256.h5', str(sys.argv[3]))
#predicted_projections.append(predictions[0].squeeze())
#print(predicted_projections[0].min(), predicted_projections[0].max())

#predicted_projections[0] = predicted_projections[0] * 10922.5
#predicted_projections[0] = predicted_projections[0].astype('uint16')
#with h5py.File(str(sys.argv[3]), 'r+') as outputFile:
#    voxelData = outputFile['ITKImage/0/VoxelData']
#    voxelData[...] = predicted_projections[0]

