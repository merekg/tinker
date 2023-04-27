import shutil
import h5py
import numpy as np
import os
def read_data(file_path):
    projections = h5py.File(file_path, 'r')['ITKImage/0/VoxelData'][:]
    return projections
def save_data(input_file_path, output_file_path, projections):
    shutil.copyfile(input_file_path, output_file_path)
    outputFile = h5py.File(output_file_path, 'r+')
    voxelData = outputFile['ITKImage/0/VoxelData']
    voxelData[...] = projections
    
    

def flip(projections, axis):
    flipped_projections = np.flip(projections, axis =axis)
    return flipped_projections

def add_noise(projections):
    noise = np.random.normal(0, 1, size = projections.shape)
    noisy_projections = projections + noise
    return noisy_projections

def augment(input_dir, output_dir):
    for file_name in os.listdir(input_dir):
        projections = read_data(input_dir+file_name)
        name, ext = file_name.split(".h5")
        
        projections = flip(projections, 0)
        save_data(input_dir+file_name, output_dir+name+'flipped_0_modified.h5', projections)
        
        projections = flip(projections, 1)
        save_data(input_dir+file_name, output_dir+name+'flipped_1_modified.h5', projections)
        
        projections = flip(projections, 2)
        save_data(input_dir+file_name, output_dir+name+'flipped_2_modified.h5', projections)
        
        projections = flip(projections, 0)
        projections = add_noise(projections)
        save_data(input_dir+file_name, output_dir+name+'flipped_noise_0_modified.h5', projections)
        
        projections = flip(projections, 1)
        projections = add_noise(projections)
        save_data(input_dir+file_name, output_dir+name+'flipped_noise_1_modified.h5', projections)
        
        projections = flip(projections, 2)
        projections = add_noise(projections)
        save_data(input_dir+file_name, output_dir+name+'flipped_noise_2_modified.h5', projections)
        
augment('/home/ubuntu/Desktop/MachineLearning/ViewerMachineLearning/DataForLabelling/', '/home/ubuntu/Desktop/MachineLearning/ViewerMachineLearning/Augmented/')
        
            

