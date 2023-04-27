import csv
import h5py
import numpy as np
import os
import time
import sys
import DataLoader

#master_orientation = []
#with open('/home/ubuntu/Desktop/MachineLearning/ViewerMachineLearning/test/Cadaver_Cervical.h5', newline = '') as csvfile:
    #spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #for row in spamreader:
        #master_orientation = row[0:]
        
for files in os.listdir('/home/ubuntu/Desktop/MachineLearning/ViewerMachineLearning/test/'):
    master_orientation = ['hello', 'I', 'S', 'L', 'R', 'P', 'A']
    indexS = master_orientation.index('S')
    indexI = master_orientation.index('I')
    indexR = master_orientation.index('R')
    indexL = master_orientation.index('L')
    indexP = master_orientation.index('P')
    indexA = master_orientation.index('A')
    if '0_modified' in files or '1_modified' in files:
        master_orientation[indexP], master_orientation[indexA] = master_orientation[indexA], master_orientation[indexP]
        master_orientation[indexL], master_orientation[indexR] = master_orientation[indexR], master_orientation[indexL]
    if '2_modified' in files:
        master_orientation[indexP], master_orientation[indexA] = master_orientation[indexA], master_orientation[indexP]
        master_orientation[indexL], master_orientation[indexR] = master_orientation[indexR], master_orientation[indexL]
        master_orientation[indexI], master_orientation[indexS] = master_orientation[indexS], master_orientation[indexI]
    with open('/home/ubuntu/Desktop/MachineLearning/ViewerMachineLearning/test_csv/engineerind-data-'+files+'.csv', 'w',newline = '') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(master_orientation)
        
    
    
        
        
        
    
