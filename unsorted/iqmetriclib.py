import numpy as np
import math
import nrrd
import os
import sys
import h5py

def correctNumberOfPoints(dataPoints):
    if len(dataPoints) == 0:
        print("No coordinates entered.")
        raise SystemExit()
    elif len(dataPoints) % 3 != 0:
        print("Need the correct number of coordinates.")
        raise SystemExit()
    else:
        print("")

def getHdf5FileList(inputFilePath):
    fileList = os.listdir(inputFilePath)
    fileList.sort(key=str)
    hdf5FileList = [i for i in fileList if i.endswith('.h5')]
    
    return hdf5FileList

def getRanges(centerPoint, radius, spacings):
    # Find approximate range of indeces for cube of 2*radius X 2*radius X 2*radius
    xMaxDist = int(math.ceil(radius/spacings[0]))
    yMaxDist = int(math.ceil(radius/spacings[1]))

    xRange = range(centerPoint[0]-xMaxDist-1,centerPoint[0]+xMaxDist+1)    #These are indices for slices
    yRange = range(centerPoint[1]-yMaxDist-1,centerPoint[1]+yMaxDist+1)
    
    if spacings[2] == 0:
        zRange = [centerPoint[2]]
    else:
        zMaxDist = int(math.ceil(radius/spacings[2]))
        zRange = range(centerPoint[2]-zMaxDist-1,centerPoint[2]+zMaxDist+1)

    return [xRange, yRange, zRange]

def getPointsSphere(center,InVolume,ranges,spacings,radiusIn,radiusOut):
    RoiPoints = []
    RoiValues = []
    RoiDist = []
    for x in ranges[0]:
        for y in ranges[1]:
            for z in ranges[2]:
                dist = np.sqrt( ( (x - center[0]) * spacings[0] )**2 + \
                                ( (y - center[1]) * spacings[1] )**2 + \
                                ( (z - center[2]) * spacings[2] )**2 )
                if radiusIn <= dist  <= radiusOut:
                    RoiPoints.append([x,y,z])
                    RoiValues.append(InVolume[x,y,z])
                    RoiDist.append(dist)
    return RoiPoints, RoiValues, RoiDist

def getDataRoiCylinder(InVolume,ranges,spacings,radiusIn,radiusOut,depth,pointOnLine,axisUnitVector):
    RoiDistAxis = []    # data point distance from axis
    RoiValues = []      # data point pixel value
    RoiPoints = []      # data point location
    tmpArray = []
    for x in ranges[0]:
        for y in ranges[1]:
            for z in ranges[2]:
                # Find distance from axis
                tmpPixelArray = np.array([x,y,z])
                tmpSpacingsArray = np.array(spacings)
                tmpArray = tmpPixelArray * tmpSpacingsArray
                lineToPoint = np.array(tmpArray - pointOnLine)
                crossProduct = np.cross(lineToPoint,axisUnitVector)
                DistAxis = np.sqrt(crossProduct.dot(crossProduct)) / np.sqrt(axisUnitVector.dot(axisUnitVector))
                # Find distance from plane
                dotProduct = np.dot(lineToPoint,axisUnitVector)
                DistPlane = abs(dotProduct) / np.sqrt(axisUnitVector.dot(axisUnitVector))
                # Compare point distance from axis and from plane
                if radiusIn <= DistAxis <= radiusOut and DistPlane < depth/2:
                    RoiPoints.append([x,y,z])
                    RoiValues.append(InVolume[x,y,z])
                    RoiDistAxis.append(DistAxis)
    # Return value as a tuple
    return RoiPoints, RoiValues, RoiDistAxis

def getDataRoiCircle(center,InVolume,ranges,spacings,radiusIn,radiusOut):
    RoiPoints = []
    RoiValues = []
    RoiDist = []
    for x in ranges[0]:
        for y in ranges[1]:
            for z in ranges[2]:
                # Assume 2D image or 2D plane
                dist = np.sqrt( ( (x - center[0])*spacings[0] )**2 + ( (y - center[1])*spacings[1] )**2 )
                if radiusIn <= dist <= radiusOut:
                    RoiPoints.append([x,y])
                    RoiValues.append(InVolume[x,y])
                    RoiDist.append(dist)
    # Return value as a tuple
    return RoiPoints, RoiValues, RoiDist

def getSlicedVerticalCylinderMask(shape, spacing, center, radius, height, angle):

    # Create measures across x, y, and z
    size = np.array(spacing) * np.array(shape)
    x = np.arange(-center[0], size[0] - center[0], spacing[0])
    y = np.arange(-center[1], size[1] - center[1], spacing[1])
    z = np.arange(-center[2], size[2] - center[2], spacing[2])

    # create mask for angle
    angle = np.pi * ((np.array(angle) - 180) / 180.0)
    angles = np.arctan2(y[None,:], -x[:,None])
    angleMask = (angle[0] < angles) & (angles < angle[1])

    # Create masks for radius and height
    radiusMask = (x[:,None]**2 + y[None,:]**2) < radius**2
    heightMask = abs(z) < height / 2.0

    # Return overlap (and) of radius, height, and angle masks
    return radiusMask[:,:,None] & heightMask[None,None,:] & angleMask[:,:,None]

# Center must be in dimensions of the spacing
def getVerticalCylinderMask(shape, spacing, center, radius, height):

    # Create measures across x, y, and z
    size = np.array(spacing) * np.array(shape)
    x = np.linspace(-center[0], size[0] - center[0], shape[0], False)
    y = np.linspace(-center[1], size[1] - center[1], shape[1], False)
    z = np.linspace(-center[2], size[2] - center[2], shape[2], False)

    # Create masks for radius and height
    radiusMask = (x[:,None]**2 + y[None,:]**2) < radius**2
    heightMask = abs(z) < height / 2.0

    # Return overlap (and) of radius and height masks
    return radiusMask[:,:,None] & heightMask[None,None,:]

def getIndexCoordinates(ranges):
    roiPoints = []
    
    if len(ranges) == 3:    # 3D, therefore 3 coordinate indices
        if len(ranges[0]) == 1:
            x = ranges[0][0]
            for y in ranges[1]:
                for z in ranges[2]:
                    roiPoints.append([x,y,z])
        elif len(ranges[0]) != 1 and len(ranges[1]) == 1:
            y = ranges[1][0]
            for x in ranges[0]:
                for z in ranges[2]:
                    roiPoints.append([x,y,z])
        elif len(ranges[0]) != 1 and len(ranges[1]) != 1 and len(ranges[2]) != 1:
            for x in ranges[0]:
                for y in ranges[1]:
                    for z in ranges[2]:
                        roiPoints.append([x,y,z])
    if len(ranges) == 2:    # 2D, therefore 2 coordinate indices
        if len(ranges[0]) == 1:
            x = ranges[0][0]
            for y in ranges[1]:
                roiPoints.append([x,y])
        elif len(ranges[0]) != 1 and len(ranges[1]) == 1:
            y = ranges[1][0]
            for x in ranges[0]:
                roiPoints.append([x,y])
        elif len(ranges[0]) != 1 and len(ranges[1]) != 1:
            for x in ranges[0]:
                for y in ranges[1]:
                    roiPoints.append([x,y])
                
    return roiPoints

def getValues(RoiPoints, Volume):
    RoiValues = []
    for slicePos in RoiPoints:
        RoiValues.append(Volume[slicePos[0],slicePos[1],slicePos[2]])
    return RoiValues

def setValues(RoiPoints, InVolume, Value):
    tmpVol = InVolume
    if len(InVolume.shape) == 3:
        for slicePos in RoiPoints:
            tmpVol[slicePos[0],slicePos[1],slicePos[2]] = Value
    elif len(InVolume.shape) == 2:
        for slicePos in RoiPoints:
            tmpVol[slicePos[0],slicePos[1]] = Value
    return tmpVol

def getMTF(xf, mtfValues, percent):

    # Find all values that cross the precent threshold
    thresholdIndices = mtfValues < percent

    # If all values below threshold return lowest MTF
    if np.all(thresholdIndices):
        return xf[0]

    # If all values above trheshold return highest MTF
    if np.all(thresholdIndices == False):
        #return xf[-1] #TODO: Take only half of FFT as input instead of dividing in half
        return xf[len(xf)/2]

    # Find the first point that crosses the percent treshold
    firstCrossing = np.argmax(thresholdIndices)

    # Find bondry points of min MTF
    y2 = mtfValues[firstCrossing]
    y1 = mtfValues[firstCrossing-1]
    x2 = xf[firstCrossing]
    x1 = xf[firstCrossing-1]

    # Linearly estimate MTF frequency 
    slopeMTF = (y2 - y1) / (x2 - x1)
    interceptMTF =  y1 - slopeMTF * x1
    freqAtPercent = (percent - interceptMTF)/slopeMTF

    return freqAtPercent

def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

# TODO: Hard coded values need to be removed and passed to the function
# TODO: Break function into two separate functions or 1 that passes needed conversion values for HU and cmInv
def convert_Pixel2HU_Pixel2cmInv(inputScale, volume_pixel):
    # inputScale is the 8192 values used to scale data from the nView system
    # if inputScale is a None, then it is assumed data is from CBCT system (i.e. O-arm)
    # Current conversion values are approximated and not measured.
    
    # Determine Linear Attenuation Coefficients for water and air for conversion
    if inputScale is None:
        rescaleSlope = 1
        rescaleIntercept = -1024
        # O-arm operates at 120 kV, therfore appr. 80 keV (1/3 of max kV)    
        u_water = 0.175    #water u [cm^-1] at 80keV
        u_air = 0     #air u [cm^-1] at 80keV
    else:  
        # nView operates at 70 kV, measured values in 2/2018 appr. 55keV
        u_water = 0.21637    #water u [cm^-1]
        u_air = 0.00002     #air u [cm^-1]

    # Rescale values for HU and cm^-1 units
    if inputScale is None:    # If O-Arm
        volume_cmInv = (volume_pixel*(u_water-u_air)/1000)+u_water    # convert to cm^-1
        volume_HU = volume_pixel*rescaleSlope + rescaleIntercept    # convert to HU
    else:    # If nView system
        volume_cmInv = volume_pixel/float(inputScale)    # convert to cm^-1
        volume_HU = ((volume_cmInv-u_water)*1000)/(u_water-u_air)    # convert to HU
        
    # Return value as a tuple
    return volume_cmInv, volume_HU

def saveOuputVolume(outputFolder, tmpVolume, tmpVolumeSpacings):    
    # Write values to nrrd file to highlight ROIs
    filenameB = (outputFolder + 'OutputVolume.nrrd')
    options = {}
    options["spacings"] = tmpVolumeSpacings
    nrrd.write(filenameB, tmpVolume, options)

def saveVolume(outputFolder, tmpVolume, tmpVolumeSpacings, tmpVolumeName):
    # Write values to nrrd file to highlight ROIs
    filenameB = (outputFolder + tmpVolumeName + '.nrrd')
    options = {}
    options["spacings"] = tmpVolumeSpacings
    nrrd.write(filenameB, tmpVolume, options)

#TODO: The file saves but does not open in MITK. When fixed can replace other save volume functions.
def saveVolumeHdf5(outputFolder, tmpVolume, tmpVolumeSpacings, tmpVolumeName):
    filenameB = (outputFolder + tmpVolumeName + '.h5')
    
    hf = h5py.File(filenameB, 'w')
    hf.create_dataset('dataset_1', data=tmpVolume)
    hf.close()
