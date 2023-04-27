import h5py
import numpy as np
import shutil
import os
import transforms

#Assign directory.
dataDirectory = '/home/tristanmary/Desktop/Shalin/MLVERTEBRAELABELLING/OriginalData/'
labelDirectory = '/home/tristanmary/Desktop/Shalin/MLVERTEBRAELABELLING/Labels/'

augmentedDataDirectory = '/home/tristanmary/Desktop/Shalin/MLVERTEBRAELABELLING/AugmentedData/'
augmentedLabelDirectory = '/home/tristanmary/Desktop/Shalin/MLVERTEBRAELABELLING/AugmentedLabels/'

#Set the seed
np.random.seed()
    
for file in os.listdir(dataDirectory):
    dataFile = dataDirectory + file
    labelFile = labelDirectory + file
    print('Addressing file', file)
    for i in range(0, 5):
        #Generate the random file name.
        augmentedFilename = np.random.randint(2000000000, 3000000000, size=1)
        print(augmentedFilename)
        augmentedFileData = augmentedDataDirectory + str(augmentedFilename[0]) + '.h5'
        if os.path.exists(augmentedFileData):
            augmentedFilename = np.random.randint(2000000000, 3000000000, size=1)
        
        augmentedFileData = augmentedDataDirectory + str(augmentedFilename[0]) + '.h5'
        augmentedFileLabel = augmentedLabelDirectory + str(augmentedFilename[0]) + '.h5'
        
        shutil.copyfile(dataFile, augmentedFileData)
        shutil.copyfile(labelFile, augmentedFileLabel)
        
        augmentedFileDataOpen = h5py.File(augmentedFileData, 'r+')
        augmentedFileLabelOpen = h5py.File(augmentedFileLabel, 'r+')
        
        dataVoxel = augmentedFileDataOpen['ITKImage/0/VoxelData']
        labelVoxel = augmentedFileLabelOpen['ITKImage/0/VoxelData']
        
        dataVoxelProjections = dataVoxel[...]
        labelVoxelProjections = labelVoxel[...]
        
        if i == 0:
            print(augmentedFilename[0], '0')
            theta   = float(np.around(np.random.uniform(-10.0,10.0, size=1), 2))
            dataVoxelProjections  = transforms.rotateit(dataVoxelProjections, theta)
            labelVoxelProjections = transforms.rotateit(labelVoxelProjections, theta, isseg=True) 

        elif i == 1:
            print(augmentedFilename[0], '1')
            scalefactor  = float(np.around(np.random.uniform(0.9, 1.1, size=1), 2))
            dataVoxelProjections  = transforms.scaleit(dataVoxelProjections, scalefactor)
            labelVoxelProjections = transforms.scaleit(labelVoxelProjections, scalefactor, isseg=True) 

        elif i == 2:
            print(augmentedFilename[0], '2')
            factor  = float(np.around(np.random.uniform(0.8, 1.2, size=1), 2))
            dataVoxelProjections  = transforms.intensifyit(dataVoxelProjections, factor)
            #no intensity change on segmentation

        elif i == 3:
            print(augmentedFilename[0], '3')
            dataVoxelProjections = transforms.noisy(dataVoxelProjections)
            

        elif i == 4:
            print(augmentedFilename[0], '4')
            offset  = list(np.random.randint(-10,10, size=3))
            currseg = labelVoxelProjections
            dataVoxelProjections  = transforms.translateit(dataVoxelProjections, offset)
            labelVoxelProjections = transforms.translateit(labelVoxelProjections, offset, isseg=True)
        
        dataVoxel[...] = dataVoxelProjections
        labelVoxel[...] = labelVoxelProjections
        print('Done Augmenting', file, i)

    
    
    
    
