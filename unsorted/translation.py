import numpy as np 
import random
from skimage.transform import resize
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

random.seed(23)

DAT_FILE_PATH = '/media/save/Shalin/SmartScrolling/Volumes/'
IMAGE_PATH = '/media/save/Shalin/SmartScrolling/Img/'

def getVolume(filePath):
	voxel = np.fromfile(filePath, dtype='uint8')
	voxelData = np.reshape(voxel, (416,416,416))
	return voxelData

dataFile = np.genfromtxt('/media/save/Shalin/SmartScrolling/CubicFiles/points.csv', delimiter=',')



def shift(volume, data, shift):
	print(shift)
	rolled = np.roll(volume, int(shift * 0.721154), axis = 1)
	resized = resize(rolled, (64, 64, 64), mode = 'constant', preserve_range = True, anti_aliasing=False).astype('uint16')
	volumeResized = np.expand_dims(resized, 3)
	volumeResized = volumeResized/255
	shiftedData = data.copy()
	shiftedData[3] = shiftedData[3] + shift
	shiftedData[7] = shiftedData[7] + shift
	shiftedData[11] = shiftedData[11] + shift
	shiftedData[15] = shiftedData[15] + shift
	return volumeResized, shiftedData


def saveImage(v, name, shift):
	projections = np.amax(v, 0)
	plt.imshow(projections)
	plt.savefig(IMAGE_PATH + str(name) + '_' + str(shift)+ '.png')

	
mainData = []
mainLabel = []
augTracker = []
for i in range(0, 56):
	label = dataFile[i][:]
	filePath = DAT_FILE_PATH + str(int(dataFile[i][1])) + '.dat'
	print(filePath);
	volume = getVolume(filePath)

	shiftedVolume, shiftedLabel = shift(volume, label, 0)
	mainData.append(shiftedVolume)
	mainLabel.append(shiftedLabel)
	# augTracker.append([filePath, 0])
	# saveImage(shiftedVolume, int(label[0]), 0)
	# for i in range (0, 4):
	# 	shift_mm = random.randrange(-40, 40)
	# 	shiftedVolume, shiftedLabel = shift(volume, label, shift_mm)
	# 	mainData.append(shiftedVolume)
	# 	mainLabel.append(shiftedLabel)
	# 	augTracker.append([filePath, shift_mm])
		# saveImage(shiftedVolume, int(label[0]), shift_mm)
	
# mainLabelnpy = np.asarray(mainLabel)
mainDatanpy = np.asarray(mainData)
# augTrackernpy = np.asarray(augTracker)

# print(mainLabelnpy.shape)
print(mainDatanpy.shape)
# np.savetxt("/media/save/Shalin/SmartScrolling/CubicFiles/augmentedCubicTrainLabel.csv", mainLabelnpy, delimiter=",", fmt='%f')
# np.savetxt("/media/save/Shalin/SmartScrolling/CubicFiles/augmentedTracker.csv", augTracker, delimiter=",", fmt='%s')
np.save('/media/save/Shalin/SmartScrolling/Data/CubicDataToTest.npy', mainDatanpy)