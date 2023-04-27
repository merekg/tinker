#!/usr/bin/env python3

import sys
import h5py
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter
from scipy.ndimage import sobel

import matplotlib.pyplot as plt

# Keys for reading and writing CSV file
CSV_FILE_KEYS = [
        'Comparison Name',
        'Playback Flags',
        'Reconstruction Flags',
        'Experiment',
        'Experiment Factorial',
        'Save Path',
        'Final L1/N',
        'Numerial Analysis',
        'Visual Analysis Component']

# HDF5 paths for reconstruction volumes
HDF5_VOLUME_PATH = 'ITKImage/0/VoxelData'
HDF5_DIM_PATH = 'ITKImage/0/Dimension'
HDF5_SCALE_ATTRIBUTE = 'scale'

def printHelp():

    print('Summary: This application will select generate a linear model based on input csv data: (l1OverN,VisualizationFactor)')
    print('Usage: python LinearRegression.py <input-csv>')

def loadExperimentList(inputCSVPath):
    try:
        reader = csv.reader(open(inputCSVPath), delimiter=';')
        comparisonList = [dict(zip(CSV_FILE_KEYS, row)) for row in list(reader)]
    except Exception as e:
        print('Failed to load input CSV file', str(e))
        printHelp()
        exit()

    return comparisonList[1:]

def loadVolume(volumePath):

    try:
        with h5py.File(volumePath, 'r') as volumeFile:
            voxelData = volumeFile[HDF5_VOLUME_PATH][:].astype(float)
            dimensions = volumeFile[HDF5_DIM_PATH][:].astype(float)
            scaleFactor = float(volumeFile[HDF5_VOLUME_PATH].attrs[HDF5_SCALE_ATTRIBUTE])
            volume = voxelData / scaleFactor
    except Exception as e:
        print('Failed to load volume: ' + volumePath + ' : ' + str(e) + '\n')
        volume = np.array([])

    return volume,dimensions

def compareEdge(volume):

    try:
        gaussianVolume = gaussian_filter(volume, sigma=1)
        sobelVolume = sobel(gaussianVolume)
        l1SobelResponse = np.linalg.norm(sobelVolume)
    except Exception as e:
        print('Failed to compute Edge comparison: ' + str(e) + '\n')
        l1SobelResponse = 0

    return l1SobelResponse

def main():
    inputCsv = sys.argv[1]
    
    comparisonVolume = None
    comparisonVolumeDimensions = None
    applyTwoFactorEffects = True
    
    if len(sys.argv) > 2:
        applyTwoFactorEffects = sys.argv[2]
    if len(sys.argv) > 3:
        comparisonVolume,comparisonVolumeDimensions = loadVolume(sys.argv[3])
        
    
    experimentList = loadExperimentList(inputCsv)
    
    X = []
    y = []
    sumResiduals = 0
    
    for experiment in experimentList:
        reconstructionVolume,volumeDim = loadVolume(experiment['Save Path'])
        
        if comparisonVolume != None:
            print("TODO")
        
        factorialList = experiment['Experiment Factorial'].split()
        
        #Main effects
        x = []
        for factor in factorialList:
            x.append(float(factor))
            
        twoFactorEffects = []
        if applyTwoFactorEffects:
            #Two-factor effects
            index = 0
            while index < (len(x) - 1):
                subIndex = index + 1
                while subIndex < len(x):
                    twoFactorEffects.append(x[index] * x[subIndex])
                    subIndex = subIndex + 1
                index = index + 1
        
        #Three-factor effects
        #index = 0
        #threeFactorEffects = []
        #while index < (len(x) - 2):
        #    subIndex = index + 1
        #    while subIndex < (len(x) - 1):
        #        tertiaryIndex = 0
        #        while tertiaryIndex < len(x):
        #            threeFactorEffects.append(x[index] * x[subIndex] * x[tertiaryIndex])
        #            tertiaryIndex = tertiaryIndex + 1
                    
        #        subIndex = subIndex + 1
        #    index = index + 1
        
        for effect in twoFactorEffects:
            x.append(effect)
        
        #for effect in threeFactorEffects:
        #    x.append(effect)
        
        #Anything beyond 3-factor effects is unlikely to be useful
        
        
        X.append(x)
        
        #mtfCoordinates = experiment['CNR Coordinates'].split('')
        #cnrCoordinates = experiment['MTF Coordinates'].split('')
        #mtf = iqm.runMTF2D(reconstructionVolume, mtfCoordinates)
        #cnr = iqm.runCNR(reconstructionVolume, cnrCoordinates)
        
        visualizationResponse = float(experiment['Visual Analysis Component'])
        l1Response = float(experiment['Final L1/N'])
        numericalAnalysis = float(experiment['Numerial Analysis'])
        
        edgeResponse = compareEdge(reconstructionVolume)
        
        response = 0.5 * (visualizationResponse - 2) + 0.3 * (1.0/l1Response) + 0.2 * (1.0/numericalAnalysis)# + 0.2 * edgeResponse #- 5*mtf #- 0.04*cnr
        
        sumResiduals = sumResiduals + response
        y.append(response)
    
    reg = LinearRegression().fit(X, y)
    
    print("Intercept ", reg.intercept_)
    
    mainCoeffs = []
    secondaryCoeffs = []
    primaryCoeffRange = len(experiment['Experiment Factorial'].split())
    
    i = 0
    while i < len(reg.coef_):
        if i < primaryCoeffRange: 
            mainCoeffs.append(reg.coef_[i])
        else:
            secondaryCoeffs.append(reg.coef_[i])
        i += 1
            
    print("Primary Coefficients: ", mainCoeffs)
    print("Secondary Coefficients: ", secondaryCoeffs)
    
    meanResidual = sumResiduals/len(y)
    resSS = 0
    totSS = 0
    
    index = 0
    while index < len(X):
        sample = [X[index]]
        yPredicted = reg.predict(sample)
        resSS = resSS + (yPredicted[0] - y[index]) ** 2
        totSS = totSS + (yPredicted[0] - meanResidual) ** 2
        index = index + 1
    rValue = 1 - resSS/totSS
    print("R^2: ", rValue)
    rValueAdjusted =  1 - (((1 - rValue) * (len(X) - 1))/(len(X) - len(X[0]) - 1))
    print("R^2 Adjusted: ", rValueAdjusted)
    
    factorialTest = ''
    mean = np.mean(reg.coef_)
    sd = np.std(reg.coef_)
    for coefficient in reg.coef_:
        if coefficient > (mean + 2 * sd):
            factorialTest = factorialTest + ' +2'
        elif coefficient > (mean + sd):
            factorialTest = factorialTest + ' +1'
        elif coefficient < (mean - 2 * sd):
            factorialTest = factorialTest + ' -2'
        elif coefficient < (mean - sd):
            factorialTest = factorialTest + ' -1'
        else:
            factorialTest = factorialTest + ' 0'
            
    print("Recommended sequential test: ", factorialTest)
    
    
        
        
if __name__ == "__main__" :
    main()

