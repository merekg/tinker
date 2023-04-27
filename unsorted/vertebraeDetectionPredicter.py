import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
from tensorflow import keras
from skimage.transform import resize
import sys
import json

filePath = sys.argv[1]
dimensions = sys.argv[2]


resolution = 64
spacing = 4.6875
threshold = 0.8

def getVolume(filePath):
	voxel = np.fromfile(filePath, dtype='uint8')
	voxelData = np.reshape(voxel, (416, 416, 416))
	return voxelData

#We suppress any noise that can skew the predictions. 
def suppress(p):
	for i in range(0, resolution):
		for j in range(0, resolution):
			for k in range(0, resolution):
				if p[i, j, k] > threshold:
					temp = p[i, j, k]
					p[i - 3 : i + 3, j - 3 : j + 3, k - 3 : k + 3 ] = 0
					p[i,j,k] = temp
	return p 

volume = getVolume(filePath)
#resize and normalize
resized = resize(volume, (resolution, resolution, resolution), mode = 'constant', preserve_range = True, anti_aliasing=False).astype('uint8')
volumeResized = np.expand_dims(resized, 3)
volumeResized = np.expand_dims(volumeResized, 0)
volumeResized = volumeResized/255.0

#Clear any backend session and load the model without compiling.
keras.backend.clear_session()
model = keras.models.load_model('/opt/rt3d/ML/vertebraeDetection.h5', compile = False)

#Predict
p = model.predict(volumeResized)

#Remove extra dimesions.
p = np.squeeze(p, 4)
p = np.squeeze(p, 0)
#Convert to binary
p = np.transpose(p, (2,1,0))
p = (p > threshold)
#Suppress noise
p = suppress(p)
#Extract the points in the volume space.
result = np.where(p >= 1)



listOfCoordinates = list(zip(result[0] * spacing, result[1] * spacing, result[2] * spacing, p[result[0], result[1], result[2]]))
#Make them in viewer understandable format.
mlPredictions = []
for cord in listOfCoordinates:
	jsonDict = {'label': '', 'location': [{'x': cord[0], 'y': cord[1], 'z': cord[2], '3': 1}]}
	y = json.dumps(jsonDict)
	mlPredictions.append(y)

d = {'vertebraePredicted' : mlPredictions}
j = json.dumps(d)
print(j)