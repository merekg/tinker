import numpy as np
import matplotlib.pyplot as plt
import h5py
import DataLoader

pred = DataLoader.read_data('/home/ubuntu/Desktop/MachineLearning/Validation/Predictions/')
gt = DataLoader.read_data('/home/ubuntu/Desktop/MachineLearning/Validation/Gt/')
recon = DataLoader.read_data('/home/ubuntu/Desktop/MachineLearning/Validation/earlyrecon')
