import h5py
import numpy as np
import os
import time
import sys

def read_files(parent_dir):
	file_list = []
	for file in os.listdir(parent_dir):
		file_list.append(parent_dir+file)
	file_list.sort()
	print(file_list)
	return file_list
def read_labels(parent_dir):
    filelist = read_files(parent_dir)
    label_list = []
    for file in filelist:
        one_hot_label = []
        volInput = h5py.File(file, 'r')
        label = volInput['ITKImage/0/VoxelData'].attrs['PatientHeadRight']
        label = label.decode()
        if label == '1':
            one_hot_label.append(1)
        elif label == '':
            one_hot_label.append(0)
        label_list.append(one_hot_label)
    targets = np.asarray(label_list)
    return targets
def read_data(parent_dir):
	filelist = read_files(parent_dir)
	dataList = []
	for file in filelist:
		volInput = h5py.File(file, 'r')
		projections = volInput['ITKImage/0/VoxelData'][:]
		#projections = projections/10922.5
		projections =  np.expand_dims(projections, axis= 3)
		dataList.append(projections)
	data = np.asarray(dataList)
	return data
def save_numpy(filepath, arr):
	np.save(filepath, arr)

if __name__ == '__main__':
	data = read_data(parent_dir)
