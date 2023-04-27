import h5py
import numpy as np
import shutil
import os

labelsFilePath = '/media/save/Shalin/MLVERTEBRAELABELLING/Labels/'

for file in os.listdir(labelsFilePath):
    label = np.load(labelsFilePath + file);
    
