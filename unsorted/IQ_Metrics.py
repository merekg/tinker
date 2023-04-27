import sys
import numpy as np
import nrrd
from scipy.integrate import simps
import matplotlib.pyplot as plt
import math
import iqmetriclib as iq
import os
import matplotlib.cm as cm
import h5py

# The default behavior is to save the debug outputs for the IQ Metrics protocol
DEBUG = True

# If main turn output on TODO: Find more elegant solution to text output in general
if __name__ == '__main__':
    OUTPUT = True
else:
    OUTPUT = False

#TODO: This needs to be read from metadata when added to HDF5, only for nView files
DEFAULT_IMAGE_SCALE = 10922.5
HDF5_VOLUME_KEY = 'ITKImage/0/VoxelData'
HDF5_VOLUME_AXIS = (2,1,0) # swap [x,y,z] to [z,y,x]
HDF5_SPACINGS_KEY = 'ITKImage/0/Spacing'
HDF5_DIMENSIONS_KEY = 'ITKImage/0/Dimension'

# Dimensions of ROIs for each test  
MAX_PIXEL_VALUE = 5000              # Max pixel for increasing brightness of output nrrd Volume

CNR_PLUG_RADIUS_mm = 3              # radius of plug cylinder for CNR test
CNR_BKGND_RADIUS_mm = 3             # radius of background cylinder for CNR test
CNR_DEPTH_mm = 10                   # length of cylinders for CNR test

# CNR values for backwards compatibility with fluoro and task specific modes
CNR_PLUG_RADIUS_INNER_mm = 0        # inner radius of center cylinder for CNR test - for CNR Catphan module
CNR_PLUG_RADIUS_OUTER_mm = 3        # inner radius of center cylinder for CNR test
CNR_BKGND_RADIUS_INNER_mm = 9       # inner radius of hollow cylinder for CNR test
CNR_BKGND_RADIUS_OUTER_mm = 12      # inner radius of hollow cylinder for CNR test

MTF_MIN_RADIUS_mm = 1               # minimum radius of hollow cylinder for MTF test
MTF_MAX_RADIUS_mm = 15              # maximum radius of hollow cylinder for MTF test
MTF_DEPTH_mm = 10                   # length of cylinders for CNR and MTF test
MTF_FILTER_WINDOW = 3               # window size for moving average filter for Edge Reponse of MTF test
MTF_THICKNESS_DIVISOR = 1.0         # would prefer 2.0 but that gives ranges where no values are found and causes issues with nan
MTF_ZPF = 7                         # zero padding factor
MTF_EDGE_ANGLE = (10, 220)          # angle spread for edge of MTF computation

UI_RADIUS_mm = 10                   # radius of spheres for Uniformity test
UI_SPACING_mm = 30                  # distance to center of peripheral spheres

NPS_ROI_DIM_X_mm = 18               # dimension for NPS ROI - this value is half the width
NPS_ROI_DIM_Y_mm = 18
NPS_ROI_DIM_Z_mm = 9               # Set to zero for a flat cube, i.e., square
NPS_BIN_SIZE_DIVISOR = 1            # would prefer 2.0 but that gives ranges where no values are found and causes issues with nan
NPS_ZPF = 8                         # zero padding factor

MTF_DEPTH_1D_mm = 10                # range of z-dimension for 1D MTF
MTF_WIDTH_1D_mm = 26                # range of x- and y-dimension to include diameter of cylinder

ZST_DEPTH_1D_mm = 20                # range of z-dimesion for Slice Thickness
ZST_PIXEL_SPREAD = 6                # range of pixels to consider for ZST
ZST_GRID_DIVISOR = 4                # determines what ratio the z-grid is shifted by for recons
ZST_GRID_SHIFT_NUM = 3              # the number to grid shifts, i.e., number of recons
ZST_BB_DIAMETER_MM = 2              # diameter of the BB used in the ZST phantom

# Location for saving files
HOME = os.path.expanduser('~')
OUTPUT_FOLDER = (HOME+'/IQ/tmp/')
if not os.path.isdir(OUTPUT_FOLDER):
    os.makedirs (OUTPUT_FOLDER)

def main():
    #TODO - Change xyzPoints to mm entry values for consistency
    script = sys.argv[0]            # Name of python script (e.g. IQ_Metrics.py)
    inputFilename = sys.argv[1]     # nrrd file path and file name (e.g. /thisFolder/thisFile.nrrd)
    iqTest     = sys.argv[2]        # IQ test to run (e.g. --cnr)
    xyzPoints = sys.argv[3:]        # Voxel location in integer index values (e.g. 215 233 39) 
    
    # Check if name of IQ test was correctly entered
    assert iqTest in ['--cnr', '--mtfFluoro', '--mtf1d', '--mtf2d', '--ui','--nps', '--nps3d', '--zst'], 'iqTest is not one of --cnr, --mtf, --ui or --nps: ' + iqTest

    # Check if correct number of points entered
    iq.correctNumberOfPoints(xyzPoints)

    # Separate fileName and filePath
    filePath, onlyFilename = os.path.split(inputFilename)
    print("\n%s" % onlyFilename)

    # Load HDF5 file
    dataset = h5py.File(inputFilename, 'r')

    # Run IQ metric
    if iqTest == '--cnr':
        print("\nRunning IQ CNR - 2D or 3D")
        runCNR(dataset, xyzPoints)

    elif iqTest == '--mtfFluoro':
        print("\nRunning IQ MTF - only 2D Volumes (i.e., AP Fluoro")
        runMTFFluoro(dataset, xyzPoints)

    elif iqTest == '--mtf1d':
        print("\nRunning IQ MTF - only 3D Volumes")
        runMTF1D(dataset, xyzPoints)

    elif iqTest == '--mtf2d':
        print("\nRunning IQ MTF - only 3D Volumes")
        runMTF2D(dataset, xyzPoints)

    elif iqTest == '--ui':
        print("\nRunning IQ UI")
        runUI(dataset, xyzPoints)

    elif iqTest == '--nps':
        print("\nRunnning IQ NPS - 2D or 3D Volumes")
        runNPS((filePath+"/"), dataset, xyzPoints)

    elif iqTest == '--nps3d':
        print("\nRunnning IQ NPS - only 3D Volumes")
        runNPS3D((filePath+"/"), dataset, xyzPoints)

    elif iqTest == '--zst':
        print("\nRunning IQ Z Slice Thickness for 1D")
        runZST((filePath+"/"), dataset, xyzPoints)

def runCNR(dataset, xyzPoints):

    volume = dataset[HDF5_VOLUME_KEY][:] # Load volume data
    volume = np.transpose(volume, HDF5_VOLUME_AXIS) # Transpose to correct coordinate frame
    imageSpacings = dataset[HDF5_SPACINGS_KEY][:]
    imageDim = dataset[HDF5_DIMENSIONS_KEY][:]
    scale = DEFAULT_IMAGE_SCALE

    # Convert coordinates from indicies to mm
    plugCenter = np.array(xyzPoints[0:3]).astype(int) * np.array(imageSpacings)
    bkgndCenter = np.array(xyzPoints[3:6]).astype(int) * np.array(imageSpacings)

    # Select plug volume
    plugMask = iq.getVerticalCylinderMask(volume.shape, imageSpacings, plugCenter, CNR_PLUG_RADIUS_mm, CNR_DEPTH_mm)
    plugValues = volume[plugMask]

    # Select background volume
    bkgndMask = iq.getVerticalCylinderMask(volume.shape, imageSpacings, bkgndCenter, CNR_BKGND_RADIUS_mm, CNR_DEPTH_mm)
    bkgndValues = volume[bkgndMask]

    # Find mean and standard deviation for plug and background
    plugMean = plugValues.mean()
    plugStd = plugValues.std()
    bkgndMean = bkgndValues.mean()
    bkgndStd = bkgndValues.std()

    # This if is for backwards compatibility of fluoro and task specific modes
    if len(xyzPoints) > 6:

        # Assign coordinates
        center = [int(xyzPoints[0]), int(xyzPoints[1]), int(xyzPoints[2])]    # Coordinate by pixel index
        centerShift = [int(xyzPoints[6]), int(xyzPoints[7]), int(xyzPoints[8])]    # Coordinate by pixel index
        volumeDepth = center[2] - centerShift[2]

        # Modify spacings and Volume depending on dimensions
        if imageDim == 3 and volumeDepth != 0:
            modifiedSpacings = imageSpacings
        elif imageDim == 3 and volumeDepth == 0:
            modifiedSpacings = [imageSpacings[0], imageSpacings[1], 0]    # change value to 0 because we are looking at a slice
        elif imageDim == 2 and volumeDepth == 0:
            modifiedSpacings = [imageSpacings[0], imageSpacings[1], 0]    # There is no z-dimension, input a 2D plane

        if imageDim == 3 and volumeDepth == 0:
            Volume = Volume[:,:,center[2]]  # Remove z-dimension to create a 2D plane with a 3D input Volume

        # Calculate unit vector of axis of cylinder for 3D Volume
        if volumeDepth != 0:
            centerArray = np.array(center)
            centerShiftArray = np.array(centerShift)
            spacingsArray = np.array(modifiedSpacings)
            pointA = centerArray * spacingsArray    # Center of plug in mm
            pointB = centerShiftArray * spacingsArray    # Point on axis in mm
            vectorAB = np.array(pointB-pointA)
            unitVecAB = vectorAB/np.sqrt(vectorAB.dot(vectorAB))    # Unit vector of axis of plug
    
        # ROI of material
        # These ranges are intentionally larger than needed to insure the ROI is contained
        cnrRanges = iq.getRanges(center, CNR_BKGND_RADIUS_OUTER_mm, modifiedSpacings)
        if volumeDepth == 0:
            allCnrPoints, allCnrValues, allCnrDist = iq.getDataRoiCircle(center,Volume,cnrRanges,modifiedSpacings,CNR_PLUG_RADIUS_INNER_mm,CNR_BKGND_RADIUS_OUTER_mm)
        elif volumeDepth != 0:
            allCnrPoints, allCnrValues, allCnrDist = iq.getDataRoiCylinder(Volume,cnrRanges,modifiedSpacings,CNR_PLUG_RADIUS_INNER_mm,CNR_BKGND_RADIUS_OUTER_mm,CNR_DEPTH_mm,pointA,unitVecAB)
    
        plugPoints = np.array(allCnrPoints)[ ( np.array(allCnrDist)>=CNR_PLUG_RADIUS_INNER_mm ) & ( np.array(allCnrDist)<CNR_PLUG_RADIUS_OUTER_mm )]
        plugValues_pixel = np.array(allCnrValues)[ ( np.array(allCnrDist)>=CNR_PLUG_RADIUS_INNER_mm ) & ( np.array(allCnrDist)<CNR_PLUG_RADIUS_OUTER_mm )]
        plugDist = np.array(allCnrDist)[ ( np.array(allCnrDist)>=CNR_PLUG_RADIUS_INNER_mm ) & ( np.array(allCnrDist)<CNR_PLUG_RADIUS_OUTER_mm )]
    
        bkgndPoints = np.array(allCnrPoints)[ ( np.array(allCnrDist)>=CNR_BKGND_RADIUS_INNER_mm ) & ( np.array(allCnrDist)<CNR_BKGND_RADIUS_OUTER_mm )]
        bkgndValues_pixel = np.array(allCnrValues)[ ( np.array(allCnrDist)>=CNR_BKGND_RADIUS_INNER_mm ) & ( np.array(allCnrDist)<CNR_BKGND_RADIUS_OUTER_mm )]
        bkgndDist = np.array(allCnrDist)[ ( np.array(allCnrDist)>=CNR_BKGND_RADIUS_INNER_mm ) & ( np.array(allCnrDist)<CNR_BKGND_RADIUS_OUTER_mm )]

        plugValues = np.array(plugValues_pixel)
        bkgndValues = np.array(bkgndValues_pixel_pixel)

        plugMean = plugValues.mean()
        plugStd = plugValues.std()
        bkgndMean = bkgndValues.mean()
        bkgndStd = bkgndValues.std()

    # Convert Pixel Intensity to cm^-1 and HU
    plugValues_cmInv, plugValues_HU = iq.convert_Pixel2HU_Pixel2cmInv(scale, plugValues)
    bkgndValues_cmInv, bkgndValues_HU = iq.convert_Pixel2HU_Pixel2cmInv(scale, bkgndValues)

    # Calculate CNR and other statistics
    SNR_Plug = plugMean / plugStd # Plug signal to noise ratio
    SNR_Bckgnd = bkgndMean / bkgndStd # Background signal to noise ratio
    SdNR = abs(plugMean - bkgndMean) / bkgndStd # Signal difference to noise ratio
    #CNR_halfDenom = abs(plugMean - bkgndMean) / np.sqrt(0.5 * (plugStd**2 + bkgndStd**2))
    CNR = abs(plugMean - bkgndMean) / np.sqrt(plugStd**2 + bkgndStd**2)

    if OUTPUT:

        # Print output to screen
        print "Plug Center:", plugCenter
        print "Background Center:", bkgndCenter
        #print "Axis point:", plugAxis

        print "\nROI data:"
        print "Mean [pixel intensity]:", np.around(plugMean, 2)
        print "StDev [pixel intensity]:", np.around(plugStd, 2)
        print "Mean [cm^-1]:", np.around(plugValues_cmInv.mean(), 4)
        print "StDev [cm^-1]:", np.around(plugValues_cmInv.std(), 4)
        print "Mean [HU]:", np.around(plugValues_HU.mean(), 2)
        print "StDev [HU]:", np.around(plugValues_HU.std(), 2)
        print "Radius [mm]:", CNR_PLUG_RADIUS_mm
        print "Number of points:", len(plugValues)

        print "\nBackground data:"
        print "Mean [pixel intensity]:", np.around(bkgndMean, 2)
        print "StDev [pixel intensity]:", np.around(bkgndStd, 2)
        print "Mean [cm^-1]:", np.around(bkgndValues_cmInv.mean(), 4)
        print "StDev [cm^-1]:", np.around(bkgndValues_cmInv.std(), 4)
        print "Mean [HU]:", np.around(bkgndValues_HU.mean(), 2)
        print "StDev [HU]:", np.around(bkgndValues_HU.std(), 2)
        print "Radius [mm]:", CNR_BKGND_RADIUS_mm
        print "Number of points:", len(bkgndValues)

        print("\nSNR plug is: %s" % np.around(SNR_Plug,2))
        print("SNR background: %s" % np.around(SNR_Bckgnd,2))
        print("SdNR is: %s" % np.around(SdNR,2))
        #print("CNR half denom is %s" % CNR_halfDenom)
        print("CNR is: %s" % np.around(CNR,2))

    # Only save outputs if debug flag is set
    if DEBUG:

        # Mark background and plug regions on volume
        volume[plugMask | bkgndMask] = volume.max()*5
        iq.saveOuputVolume(OUTPUT_FOLDER, volume, imageSpacings)
        iq.saveVolumeHdf5(OUTPUT_FOLDER, volume, imageSpacings, 'OutputVolume')

    return CNR

def runMTFFluoro(Volume, imageSpacings, scale, xyzPoints):
    
    ##For testing when now 2D projection is available
    #Volume = Volume[:,:,58]
    
    # Assign coordinates
    center = [int(xyzPoints[0]), int(xyzPoints[1]), int(xyzPoints[2])]    # Coordinate by pixel index

    # Get dimensions of entire nrrd volume
    modifiedSpacings = [imageSpacings[0], imageSpacings[1], 0]    # There is no z-dimension, input a 2D plane

    # Use spacing to determine x-axis scaling
    thickness = modifiedSpacings[0]

    # -------Edge Profile for 1D MTF
    
    # Get ranges of x and y that include plug
    ranges = iq.getRanges(center, MTF_MAX_RADIUS_mm, modifiedSpacings)

    # Get range for x and y line that are within MTF_MAX_RADIUS_mm
    for ind in range(0,len(ranges[0])):
        dist = abs( (ranges[0][ind] - center[0]) * modifiedSpacings[0] )
        print(dist)
        if dist > MTF_MAX_RADIUS_mm:
            xMaxInd = ind - 1
            break
    
    for ind in range(0,len(ranges[1])):
        dist = abs( (ranges[1][ind] - center[1]) * modifiedSpacings[1] )
        if dist > MTF_MAX_RADIUS_mm:
            yMaxInd = ind - 1
            break
    
    # line of interest
    xLoi = Volume[center[0]:ranges[0][xMaxInd],center[1]]
    yLoi = Volume[center[0],center[1]:ranges[1][yMaxInd]]

    # Only half of profile is needed for ER, flip to go from low to high
    xAvgProfile_pixel = np.flipud(xLoi)
    yAvgProfile_pixel = np.flipud(yLoi)
    
    # Convert Pixel Intensity to cm^-1 and HU
    xAvgProfile_cmInv, xAvgProfiles_HU = iq.convert_Pixel2HU_Pixel2cmInv(scale, xAvgProfile_pixel)
    yAvgProfile_cmInv, yAvgProfiles_HU = iq.convert_Pixel2HU_Pixel2cmInv(scale, yAvgProfile_pixel)
    
    # Plot of Edge profile along x-, and y-axis
    plt.figure(2)
    xAxisProfile_mm = np.arange(thickness/2, len(xAvgProfile_cmInv)*thickness,thickness)                #  CHOOSE UNITS [cm^-1]
    plt.plot(xAxisProfile_mm,xAvgProfile_cmInv, linestyle='-', marker='o',label='X Edge Profile')
    plt.plot(xAxisProfile_mm,yAvgProfile_cmInv, linestyle='--', marker='o',label='Y Edge Profile')
    plt.xlabel('distance (mm)')
    plt.ylabel('Intensity Profile [cm^-1]')
    plt.legend(loc='best')
    plt.savefig(OUTPUT_FOLDER + 'xyEdgeProfilefig')

    # Use Edge profile to find ER, LSF, and MTF
    lineProfileData = (xAvgProfile_cmInv, yAvgProfile_cmInv)
    for ind in range(0,len(lineProfileData)):
        
        lineData = lineProfileData[ind]
        if ind == 0:
            title = 'MTFx'
        elif ind == 1:
            title = 'MTFy'
        else:
            title = 'Line Profile UNKNOWN'
        
        #---------EDGE RESPONSE
        tmpEdgeResp = lineData     #for 1D yMTF
    
        edgeResp = []
        for x in tmpEdgeResp:
            edgeResp.append(float(x))
        #edgeResp = iq.movingaverage(edgeResp,MTF_FILTER_WINDOW)
        np.savetxt((OUTPUT_FOLDER + title + 'edgeReponseAvgs.out'), edgeResp)

        # X-axis in the space domain
        xNoMvgAvgFilt_mm = np.arange(thickness/2, len(edgeResp)*thickness,thickness)
        xER_mm = np.arange(thickness/2, len(edgeResp)*thickness,thickness) + thickness    # Add the shift of one dx due to moving average filter
        np.savetxt((OUTPUT_FOLDER + title + 'edgeResponseXaxis.out'), xER_mm)

        #---------LINE SPREAD FUNCTION
        # Line spread function = derivatice of Edge response
        lineSpreadFxn= np.gradient(edgeResp)
        np.savetxt((OUTPUT_FOLDER + title + 'lineSpreadFxnNoPadding.out'), lineSpreadFxn)
        
        # Zero padding to the LSF of 2**7 (i.e. 128 zeros)
        lineSpreadFxn = np.pad(lineSpreadFxn, (0,2**MTF_ZPF),'constant',constant_values=(0,0))
        #lineSpreadFxn = np.pad(lineSpreadFxn[range(1,len(lineSpreadFxn)-1)], (0,2**MTF_ZPF),'constant',constant_values=(0,0))
        np.savetxt((OUTPUT_FOLDER + title + 'lineSpreadFxn.out'), lineSpreadFxn)

        xLSF_mm = np.arange(thickness/2, len(lineSpreadFxn)*thickness,thickness) + thickness + (thickness/2)      # shift data half a step of the xER_mm to align slope with data points
        np.savetxt((OUTPUT_FOLDER + title + 'lineSpreadFxnXaxis.out'), xLSF_mm)

        # Area under line spread function, can be used for normalization
        y_values = lineSpreadFxn
        area = simps(y_values, dx=1.0)    #using the Simpson's rule for the area under a curve, interval between points is approx. 1mm

        #---------MODULATION TRANFER FUNCTION
        # X-axis in spatial frequency domain
        N = len(lineSpreadFxn)
        ts = thickness
        fs = 1/float(thickness)
        df = fs/N
        xf = np.array(range(0,N,1)) * df

        # Modulation transfer function = FFT of Line spread fxn
        tmpMtf = np.fft.fft(lineSpreadFxn)
        moduloMtf = abs(tmpMtf)
        mtfValues = moduloMtf/moduloMtf[0]
        np.savetxt((OUTPUT_FOLDER + title + 'mtfValues.out'), mtfValues)
        np.savetxt((OUTPUT_FOLDER + title + 'mtfxf.out'), xf)

        # Approximate the spatial frequency value when MTF is 50%
        percent = 50.0/100
        xfAtPercent = iq.getMTF(xf, mtfValues, percent)
        print("\nSpatial frequency at 0.5{} [mm^-1]: {:.3f}".format(title,xfAtPercent))

        ## Approximate the spatial frequency value when MTF is 10%
        #percent = 10.0/100
        #xfAtPercent = iq.getMTF(xf, mtfValues, percent)
        #print("Spatial frequency at 0.1MTF [mm^-1]: %s" % np.around(xfAtPercent,3))
        
        #---------Save and display results

        # Plot Edge Response, Line Spread Function, and MTF
        fig, axes = plt.subplots(nrows=3, figsize=(8, 10))
        axes[0].plot(xER_mm,edgeResp, linestyle='-', marker='o')
        axes[1].plot(xLSF_mm,lineSpreadFxn, linestyle='-', marker='o')
        axes[2].plot(xf[0:np.argmin(mtfValues)],mtfValues[0:np.argmin(mtfValues)], linestyle='-', marker='o')
        axes[2].plot(xfAtPercent, percent,'ro')

        axes[0].set_title(title)
        axes[0].set_ylabel('Edge Response')    #Edge Response
        axes[0].set_xlabel('distance (mm)') 
        axes[1].set_ylabel('Line Spread Function')    #Line spred function
        axes[1].set_xlabel('distance (mm)')
        axes[2].set_ylabel('MTF')
        axes[2].set_xlabel('Spatial Frequency (mm^-1)')
        axes[2].text(0.5*plt.xlim()[1],0.75*plt.ylim()[1],'50% MTF = ' + str(np.around(xfAtPercent,3)) + ' [mm^-1]')

        plt.savefig(OUTPUT_FOLDER + title + 'myfig')
        
        
    # Update nrrd volume to include cylinders
    xSlicePoints = iq.getIndexCoordinates([range(center[0],ranges[0][xMaxInd]),[center[1]]])
    ySlicePoints = iq.getIndexCoordinates([[center[0]],range(center[1],ranges[1][yMaxInd])])
    outVolume = iq.setValues(xSlicePoints, Volume, math.ceil(Volume.max()))
    outVolume = iq.setValues(ySlicePoints, Volume, math.ceil(Volume.max()))
    
    # Add ROIs to Volume
    iq.saveOuputVolume(OUTPUT_FOLDER, outVolume, [imageSpacings[0], imageSpacings[1]])
    
    iq.saveVolumeHdf5(OUTPUT_FOLDER, outVolume, imageSpacings,'OutputVolume')
    
    #plt.show()

def runMTF1D(dataset, xyzPoints):
 
    Volume = dataset[HDF5_VOLUME_KEY][:] # Load volume data
    Volume = np.transpose(Volume, HDF5_VOLUME_AXIS) # Transpose to correct coordinate frame
    imageSpacings = dataset[HDF5_SPACINGS_KEY][:]
    scale = DEFAULT_IMAGE_SCALE

    # Assign coordinates
    center = [int(xyzPoints[0]), int(xyzPoints[1]), int(xyzPoints[2])]    # Coordinate by pixel index

    # Get dimensions of entire nrrd volume
    spacings = imageSpacings

    # Use spacing to determine x-axis scaling
    thickness = spacings[0]

    # -------Edge Profile for 1D MTF
    
    # Get volume range for 1D MTF
    xRange = range( center[0] - int(math.floor(MTF_WIDTH_1D_mm/2.0/spacings[0])), center[0] + int(math.ceil(MTF_WIDTH_1D_mm/2.0/spacings[0])) )
    yRange = range( center[1] - int(math.floor(MTF_WIDTH_1D_mm/2.0/spacings[1])), center[1] + int(math.ceil(MTF_WIDTH_1D_mm/2.0/spacings[1])) )
    zRange = range( center[2] - int(math.floor(MTF_DEPTH_1D_mm/2.0/spacings[2])), center[2] + int(math.ceil(MTF_DEPTH_1D_mm/2.0/spacings[2])) )
    ranges = [xRange, yRange, zRange]

    # Get Volume of Interest (voi) from ranges
    voi = Volume[ranges[0],:,:]
    voi = voi[:,ranges[1],:]
    voi = voi[:,:,ranges[2]]
    
    # Get profile along x-axis and y-axis through center of plug
    middleIndexRange = float(len(voi[0,:,:]))/2     #assume x and y dimensions are the same
    if middleIndexRange % 1 != 0:
        midIndex = int(middleIndexRange-0.5)                    #if odd
        xAvgProfile_pixel = np.mean(voi[:,midIndex,:],axis=1)   #Average of values along the z-depth and along the x-axis
        yAvgProfile_pixel = np.mean(voi[midIndex,:,:],axis=1)
    else:
        midIndex = int(middleIndexRange)                        #if even
        xMidHigh = np.mean(voi[:,midIndex,:],axis=1)            #Average of values along the z-depth and along the x-axis
        xMidLow = np.mean(voi[:,(midIndex-1),:],axis=1)
        xAvgProfile_pixel = (xMidHigh + xMidLow) / 2
        
        yMidHigh = np.mean(voi[midIndex,:,:],axis=1)
        yMidLow = np.mean(voi[(midIndex-1),:,:],axis=1)
        yAvgProfile_pixel = (yMidHigh + yMidLow) / 2

    # Only half of profile is needed for ER
    xHalfAvgProfile_pixel = np.flipud(xAvgProfile_pixel[:midIndex])
    yHalfAvgProfile_pixel = np.flipud(yAvgProfile_pixel[:midIndex])
    
    # Convert Pixel Intensity to cm^-1 and HU
    xHalfAvgProfile_cmInv, xHalfAvgProfiles_HU = iq.convert_Pixel2HU_Pixel2cmInv(scale, xHalfAvgProfile_pixel)
    yHalfAvgProfile_cmInv, yHalfAvgProfiles_HU = iq.convert_Pixel2HU_Pixel2cmInv(scale, yHalfAvgProfile_pixel)
    
    # Plot of Edge profile along x-, and y-axis
    plt.figure(2)
    xAxisProfile_mm = np.arange(thickness/2, len(xHalfAvgProfile_cmInv)*thickness,thickness)                    #  CHOOSE UNITS [cm^-1]
    plt.plot(xAxisProfile_mm,xHalfAvgProfile_cmInv, linestyle='-', marker='o',label='X Edge Profile')
    plt.plot(xAxisProfile_mm,yHalfAvgProfile_cmInv, linestyle='--', marker='o',label='Y Edge Profile')
    plt.xlabel('distance (mm)')
    plt.ylabel('Intensity Profile [cm^-1]')
    plt.legend(loc='best')
    plt.savefig(OUTPUT_FOLDER + 'xyEdgeProfilefig')

    # Use Edge profile to find ER, LSF, and MTF
    lineProfileData = (xHalfAvgProfile_cmInv, yHalfAvgProfile_cmInv)
    for ind in range(0,len(lineProfileData)):
        
        lineData = lineProfileData[ind]
        if ind == 0:
            title = 'MTFx'
        elif ind == 1:
            title = 'MTFy'
        else:
            title = 'Line Profile UNKNOWN'
        
        #---------EDGE RESPONSE
        tmpEdgeResp = lineData     #for 1D yMTF
    
        edgeResp = []
        for x in tmpEdgeResp:
            edgeResp.append(float(x))
        edgeRespNoMvgAvgFilt = edgeResp
        #edgeResp = iq.movingaverage(edgeResp,MTF_FILTER_WINDOW)
        np.savetxt((OUTPUT_FOLDER + title + 'edgeReponseAvgs.out'), edgeResp)

        # X-axis in the space domain
        xNoMvgAvgFilt_mm = np.arange(thickness/2, len(edgeRespNoMvgAvgFilt)*thickness,thickness)
        xER_mm = np.arange(thickness/2, len(edgeResp)*thickness,thickness) + thickness    # Add the shift of one dx due to moving average filter
        np.savetxt((OUTPUT_FOLDER + title + 'edgeResponseXaxis.out'), xER_mm)

        #---------LINE SPREAD FUNCTION
        # Line spread function = derivatice of Edge response
        lineSpreadFxn= np.gradient(edgeResp)
        np.savetxt((OUTPUT_FOLDER + title + 'lineSpreadFxnNoPadding.out'), lineSpreadFxn)
        
        # Zero padding to the LSF of 2**7 (i.e. 128 zeros)
        lineSpreadFxn = np.pad(lineSpreadFxn, (0,2**MTF_ZPF),'constant',constant_values=(0,0))
        #lineSpreadFxn = np.pad(lineSpreadFxn[range(1,len(lineSpreadFxn)-1)], (0,2**MTF_ZPF),'constant',constant_values=(0,0))
        np.savetxt((OUTPUT_FOLDER + title + 'lineSpreadFxn.out'), lineSpreadFxn)

        xLSF_mm = np.arange(thickness/2, len(lineSpreadFxn)*thickness,thickness) + thickness + (thickness/2)      # shift data half a step of the xER_mm to align slope with data points
        np.savetxt((OUTPUT_FOLDER + title + 'lineSpreadFxnXaxis.out'), xLSF_mm)

        # Area under line spread function, can be used for normalization
        y_values = lineSpreadFxn
        area = simps(y_values, dx=1.0)    #using the Simpson's rule for the area under a curve, interval between points is approx. 1mm

        #---------MODULATION TRANFER FUNCTION
        # X-axis in spatial frequency domain
        N = len(lineSpreadFxn)
        ts = thickness
        fs = 1/float(thickness)
        df = fs/N
        xf = np.array(range(0,N,1)) * df

        # Modulation transfer function = FFT of Line spread fxn
        tmpMtf = np.fft.fft(lineSpreadFxn)
        moduloMtf = abs(tmpMtf)
        mtfValues = moduloMtf/moduloMtf[0]
        np.savetxt((OUTPUT_FOLDER + title + 'mtfValues.out'), mtfValues)
        np.savetxt((OUTPUT_FOLDER + title + 'mtfxf.out'), xf)

        # Approximate the spatial frequency value when MTF is 50%
        percent = 50.0/100
        xfAtPercent = iq.getMTF(xf, mtfValues, percent)

        if OUTPUT:
            print("\nSpatial frequency at 0.5{} [mm^-1]: {:.3f}".format(title,xfAtPercent))

        ## Approximate the spatial frequency value when MTF is 10%
        #percent = 10.0/100
        #xfAtPercent = iq.getMTF(xf, mtfValues, percent)
        #print("Spatial frequency at 0.1MTF [mm^-1]: %s" % np.around(xfAtPercent,3))
        
        #---------Save and display results

        # Only plot figures if debug flag is set
        if DEBUG:

            # Plot Edge Response, Line Spread Function, and MTF
            fig, axes = plt.subplots(nrows=3, figsize=(8, 10))
            axes[0].plot(xER_mm,edgeResp, linestyle='-', marker='o')
            axes[1].plot(xLSF_mm,lineSpreadFxn, linestyle='-', marker='o')
            axes[2].plot(xf[0:np.argmin(mtfValues)],mtfValues[0:np.argmin(mtfValues)], linestyle='-', marker='o')
            axes[2].plot(xfAtPercent, percent,'ro')

            axes[0].set_title(title)
            axes[0].set_ylabel('Edge Response')    #Edge Response
            axes[0].set_xlabel('distance (mm)') 
            axes[1].set_ylabel('Line Spread Function')    #Line spred function
            axes[1].set_xlabel('distance (mm)')
            axes[2].set_ylabel('MTF')
            axes[2].set_xlabel('Spatial Frequency (mm^-1)')
            axes[2].text(0.5*plt.xlim()[1],0.75*plt.ylim()[1],'50% MTF = ' + str(np.around(xfAtPercent,3)) + ' [mm^-1]')

            plt.savefig(OUTPUT_FOLDER + title + 'myfig')

    # Only save outputs if debug flag is set
    if DEBUG:

        # Update nrrd volume to include cylinders
        xSlicePoints = iq.getIndexCoordinates([ranges[0][midIndex:midIndex+1],ranges[1][:midIndex],ranges[2]])
        ySlicePoints = iq.getIndexCoordinates([ranges[0][:midIndex],ranges[1][midIndex:midIndex+1],ranges[2]])
        outVolume = iq.setValues(xSlicePoints, Volume, math.ceil(Volume.max()))
        outVolume = iq.setValues(ySlicePoints, Volume, math.ceil(Volume.max()))
        
        # Add ROIs to Volume
        iq.saveOuputVolume(OUTPUT_FOLDER, outVolume, spacings)
        iq.saveVolumeHdf5(OUTPUT_FOLDER, outVolume, imageSpacings,'OutputVolume')

    volume = dataset[HDF5_VOLUME_KEY][:] # Load volume data
    volume = np.transpose(volume, HDF5_VOLUME_AXIS) # Transpose to correct coordinate frame
    imageSpacings = dataset[HDF5_SPACINGS_KEY][:]
    scale = DEFAULT_IMAGE_SCALE

def runMTF2D(dataset, xyzPoints):

    volume = dataset[HDF5_VOLUME_KEY][:] # Load volume data
    volume = np.transpose(volume, HDF5_VOLUME_AXIS) # Transpose to correct coordinate frame
    imageSpacings = dataset[HDF5_SPACINGS_KEY][:]
    scale = DEFAULT_IMAGE_SCALE

    # For backwards compatiility with task-specific mode
    if len(xyzPoints) > 3:
    
        # Assign coordinates
        center = [int(xyzPoints[0]), int(xyzPoints[1]), int(xyzPoints[2])]    # Coordinate by pixel index
        centerShift = [int(xyzPoints[3]), int(xyzPoints[4]), int(xyzPoints[5])]    # Coordinate by pixel index

        # Get dimensions of entire nrrd volume
        spacings = imageSpacings
    
        # Calculate unit vector of axis of cylinder
        centerArray = np.array(center)
        centerShiftArray = np.array(centerShift)
        spacingsArray = np.array(spacings)
        pointA = centerArray * spacingsArray    # Center of plug in mm
        pointB = centerShiftArray * spacingsArray    # Point on axis in mm
        vectorAB = np.array(pointB-pointA)
        unitVecAB = vectorAB/np.sqrt(vectorAB.dot(vectorAB))    # Unit vector of axis of plug

        # ROI of material
        ranges = iq.getRanges(center, MTF_MAX_RADIUS_mm, spacings)
        allCylPoints, allCylValues, allCylDist = iq.getDataRoiCylinder(volume,ranges,spacings,MTF_MIN_RADIUS_mm,MTF_MAX_RADIUS_mm,MTF_DEPTH_mm,pointA,unitVecAB)

    # Use angled method to remove edge of module from computation
    else:

        edgeCenter = np.array(xyzPoints[0:3]).astype(int) * np.array(imageSpacings)
        spacings = imageSpacings

        # Select angled cylinder
        edgeMask = iq.getSlicedVerticalCylinderMask(volume.shape, imageSpacings,
                edgeCenter, MTF_MAX_RADIUS_mm, MTF_DEPTH_mm, MTF_EDGE_ANGLE)
        edgeValues = volume[edgeMask]

        # Formulate voxel distances
        size = np.array(imageSpacings) * np.array(volume.shape)
        x = np.arange(-edgeCenter[0], size[0] - edgeCenter[0], imageSpacings[0])
        y = np.arange(-edgeCenter[1], size[1] - edgeCenter[1], imageSpacings[1])
        distanceGrid = np.sqrt(x[:,None]**2 + y[None,:]**2)
        distanceGrid = np.repeat(distanceGrid[:,:,None], volume.shape[2], axis=2)
        distances = distanceGrid[edgeMask]

        allCylDist = distances
        allCylValues = edgeValues

        if DEBUG:
            volume[edgeMask] = np.max(volume) * 5
            iq.saveOuputVolume(OUTPUT_FOLDER, volume, imageSpacings)

    # Prepare lists for results of each cylinder
    ringAverage_pixel = []
    ringStdev_pixel = []
    ringAverage_cmInv = []
    ringStdev_cmInv = []
    ringAverage_HU = []
    ringStdev_HU = []

    # Get data for each cylinder
    thickness = spacings[0] / MTF_THICKNESS_DIVISOR
    counter = 0
    for m in np.arange(MTF_MIN_RADIUS_mm, MTF_MAX_RADIUS_mm, thickness):
        # Set inner and outer radius
        if m > max(allCylDist):
            break
        else:
            tmpRadiusBkgndInner = m
            tmpRadiusBkgndOuter = m+thickness
        
        # Ensure outer radius is not beyond max radius
        if tmpRadiusBkgndOuter > MTF_MAX_RADIUS_mm:
            tmpRadiusBkgndOuter = MTF_MAX_RADIUS_mm

        # ROI of material
        if len(xyzPoints) > 3:
            tmpHollowCylPoints = np.array(allCylPoints)[ ( np.array(allCylDist)>=tmpRadiusBkgndInner ) & ( np.array(allCylDist)<tmpRadiusBkgndOuter )]
            tmpHollowCylValues_pixel = np.array(allCylValues)[ ( np.array(allCylDist)>=tmpRadiusBkgndInner ) & ( np.array(allCylDist)<tmpRadiusBkgndOuter )]
        else:
            tmpHollowCylValues_pixel = allCylValues[(allCylDist >= tmpRadiusBkgndInner) & (allCylDist < tmpRadiusBkgndOuter)]
        
        # Convert Pixel Intensity to cm^-1 and HU
        tmpHollowCylValues_cmInv, tmpHollowCylValues_HU = iq.convert_Pixel2HU_Pixel2cmInv(scale, tmpHollowCylValues_pixel)

        # Calculate averages and stdev
        hollowCylAverage_pixel = np.array(tmpHollowCylValues_pixel).mean()
        hollowCylStdev_pixel = np.array(tmpHollowCylValues_pixel).std()
        hollowCylAverage_cmInv = tmpHollowCylValues_cmInv.mean()
        hollowCylStdev_cmInv = tmpHollowCylValues_cmInv.std()
        hollowCylAverage_HU = tmpHollowCylValues_HU.mean()
        hollowCylStdev_HU = tmpHollowCylValues_HU.std()

        # Only print to termianl if output enabled
        if OUTPUT:

            # Print output to screen
            print("\nROI data: %s" % m)
            print("Mean [pixel intensity]: %s" % hollowCylAverage_pixel)
            print("StDev [pixel intensity]: %s" % hollowCylStdev_pixel)
            print("Mean [cm^-1]: %s" % hollowCylAverage_cmInv)
            print("StDev [cm^-1]: %s" % hollowCylStdev_cmInv)
            print("Mean [HU]: %s" % hollowCylAverage_HU)
            print("StDev [HU]: %s" % hollowCylStdev_HU)
            print("Inner radius [mm]: %s" % tmpRadiusBkgndInner)
            print("Outer radius [mm]: %s" % tmpRadiusBkgndOuter)
            print("Cylinder length [mm]: %s" % MTF_DEPTH_mm)
            #print("Number of points: %s" % len(tmpHollowCylPoints))

        # Save values for MTF processing
        ringAverage_pixel.append(hollowCylAverage_pixel)
        ringStdev_pixel.append(hollowCylStdev_pixel)
        ringAverage_cmInv.append(hollowCylAverage_cmInv)
        ringStdev_cmInv.append(hollowCylStdev_cmInv)
        ringAverage_HU.append(hollowCylAverage_HU)
        ringStdev_HU.append(hollowCylStdev_HU)

        # Update nrrd volume to include cylinders
        #outVolume = iq.setValues(tmpHollowCylPoints, Volume, math.ceil(Volume.max()*(m/2.0 + 1)))
        #counter = counter + 1
        #if counter % 2 == 0:
            #outVolume = iq.setValues(tmpHollowCylPoints, volume, math.ceil(volume.max()*3))
        #else:
            #outVolume = iq.setValues(tmpHollowCylPoints, volume, math.ceil(volume.max()*5))
        

    #---------EDGE RESPONSE
    # Edge response data, preferrably ascending
    ringAverage_cmInv.reverse()                                                     #  CHOOSE UNITS [cm^-1]
    tmpEdgeResp = ringAverage_cmInv
    edgeResp = []
    for x in tmpEdgeResp:
        edgeResp.append(float(x))
    #edgeResp = iq.movingaverage(edgeResp,MTF_FILTER_WINDOW)

    # X-axis in the space domain
    xNoMvgAvgFilt_mm = np.arange(thickness/2, len(edgeResp)*thickness,thickness)
    xER_mm = np.arange(thickness/2, len(edgeResp)*thickness,thickness) + thickness    # Add the shift of one dx due to moving average filter

    #---------LINE SPREAD FUNCTION
    # Line spread function = derivatice of Edge response
    #lineSpreadFxn= np.gradient(edgeResp)
    lineSpreadFxn= np.diff(edgeResp)
    #lineSpreadFxn= edgeResp
    
    # Zero padding to the LSF of 2**7 (i.e. 128 zeros)
    lineSpreadFxn = np.pad(lineSpreadFxn, (0,2**MTF_ZPF),'constant',constant_values=(0,0))
    #lineSpreadFxn = np.pad(lineSpreadFxn[range(1,len(lineSpreadFxn)-1)], (0,2**MTF_ZPF),'constant',constant_values=(0,0))

    xLSF_mm = np.arange(thickness/2, len(lineSpreadFxn)*thickness,thickness) + thickness + (thickness/2)      # shift data half a step of the xER_mm to align slope with data points

    #---------MODULATION TRANFER FUNCTION
    # X-axis in spatial frequency domain
    N = len(lineSpreadFxn)
    ts = thickness
    fs = 1/float(thickness)
    df = fs/N
    xf = np.array(range(0,N,1)) * df

    # Modulation transfer function = FFT of Line spread fxn
    tmpMtf = np.fft.fft(lineSpreadFxn)
    moduloMtf = abs(tmpMtf)
    mtfValues = moduloMtf/moduloMtf[0]

    # Approximate the spatial frequency value when MTF is 50%
    percent50 = 50.0/100
    xfAtPercent50 = iq.getMTF(xf, mtfValues, percent50)

    # Approximate the spatial frequency value when MTF is 10%
    percent10 = 10.0/100
    xfAtPercent10 = iq.getMTF(xf, mtfValues, percent10)

    # Only print to terminal if output is enabled
    if OUTPUT:

        print("\nSpatial frequency at 0.5MTF [mm^-1]: %s" % np.around(xfAtPercent50,3))
        print("Spatial frequency at 0.1MTF [mm^-1]: %s" % np.around(xfAtPercent10,3))
    
    # Only ad ROIs if debug flag
    if DEBUG:

        # Add ROIs to Volume
        #iq.saveOuputVolume(OUTPUT_FOLDER, outVolume, spacings)
        #iq.saveVolumeHdf5(OUTPUT_FOLDER, outVolume, imageSpacings,'OutputVolume')

        # Plot Edge Response, Line Spread Function, and MTF
        fig, axes = plt.subplots(nrows=3, figsize=(8, 10))
        axes[0].plot(xER_mm,edgeResp, linestyle='-', marker='o')
        axes[1].plot(xLSF_mm,lineSpreadFxn, linestyle='-', marker='o')
        axes[2].plot(xf[0:np.argmin(mtfValues)],mtfValues[0:np.argmin(mtfValues)], linestyle='-', marker='o')
        axes[2].plot(xfAtPercent50, percent50,'ro')
        axes[2].plot(xfAtPercent10, percent10,'mo')

        axes[0].set_ylabel('Edge Response')    #Edge Response
        axes[0].set_xlabel('distance (mm)')
        axes[1].set_ylabel('Line Spread Function')    #Line spred function
        axes[1].set_xlabel('distance (mm)')
        axes[2].set_ylabel('MTF')
        axes[2].set_xlabel('Spatial Frequency (mm^-1)')
        axes[2].text(0.5*plt.xlim()[1],0.75*plt.ylim()[1],'50% MTF = ' + str(np.around(xfAtPercent50,3)) + ' [mm^-1]',{'color': 'r'})
        axes[2].text(0.5*plt.xlim()[1],0.65*plt.ylim()[1],'10% MTF = ' + str(np.around(xfAtPercent10,3)) + ' [mm^-1]',{'color': 'm'})

        plt.savefig(OUTPUT_FOLDER + 'myfig')

        #plt.show() 

    return xfAtPercent50 

def runUI(dataset, xyzPoints):

    # Get data from dataset
    Volume = dataset[HDF5_VOLUME_KEY][:]
    Volume = np.transpose(Volume, HDF5_VOLUME_AXIS) # Transpose to correct coordinate frame
    imageSpacings = dataset[HDF5_SPACINGS_KEY][:]
    imageDim = len(imageSpacings)
    scale = DEFAULT_IMAGE_SCALE

    # Assign coordinates
    if len(xyzPoints) > 3:
        center = [int(xyzPoints[0]), int(xyzPoints[1]), int(xyzPoints[2])]    # Coordinate by pixel index
        leftCenter = [int(xyzPoints[3]), int(xyzPoints[4]), int(xyzPoints[5])]
        rightCenter = [int(xyzPoints[6]), int(xyzPoints[7]), int(xyzPoints[8])]
        bottomCenter = [int(xyzPoints[9]), int(xyzPoints[10]), int(xyzPoints[11])]
        topCenter = [int(xyzPoints[12]), int(xyzPoints[13]), int(xyzPoints[14])]
    else:
        shiftX = int(UI_SPACING_mm / imageSpacings[0])
        shiftY = int(UI_SPACING_mm / imageSpacings[1])
        center = np.array(xyzPoints[0:3]).astype(int)
        leftCenter = center + np.array([-shiftX, 0, 0])
        rightCenter = center + np.array([shiftX, 0, 0])
        bottomCenter = center + np.array([0, -shiftY, 0])
        topCenter = center + np.array([0, shiftY, 0])

    pointList = [center, leftCenter, rightCenter, bottomCenter, topCenter]

    # Get dimensions of entire nrrd volume
    if imageDim == 3:
        modifiedSpacings = imageSpacings
    elif imageDim == 2:
        modifiedSpacings = [imageSpacings[0], imageSpacings[1], 0]    # There is no z-dimension, input a 2D plane

    # Prepare lists for results of each cylinder
    allSphereAverage_pixel = []
    allSphereStdev_pixel = []
    allSphereAverage_cmInv = []
    allSphereStdev_cmInv = []
    allSphereAverage_HU = []
    allSphereStdev_HU = []

    # Get data for each sphere
    for point in pointList:
        # ROI of material
        ranges = iq.getRanges(point, UI_RADIUS_mm, modifiedSpacings)
        if imageDim == 3:
            # ROI is a sphere
            roiPoints, roiValues, roiDist = iq.getPointsSphere(point,Volume,ranges,modifiedSpacings,0,UI_RADIUS_mm)
            roiValues_pixel = roiValues
        elif imageDim == 2:
            # ROI is a circle
            roiPoints, roiValues, roiDist = iq.getDataRoiCircle(point,Volume,ranges,modifiedSpacings,0,UI_RADIUS_mm)
            roiValues_pixel = roiValues

        # Convert Pixel Intensity to cm^-1 and HU
        sphereValues_cmInv, sphereValues_HU = iq.convert_Pixel2HU_Pixel2cmInv(scale, np.array(roiValues_pixel))
              
        # Calculate averages and stdev
        average_pixel = np.array(roiValues_pixel).mean()
        stdev_pixel = np.array(roiValues_pixel).std()
        average_cmInv = sphereValues_cmInv.mean()
        stdev_cmInv = sphereValues_cmInv.std()
        average_HU = sphereValues_HU.mean()
        stdev_HU = sphereValues_HU.std()

        # Print output to screen
        print("\nROI data:")
        print("Sphere center: %s" % point)
        print("Mean [pixel intensity]: %s" % average_pixel)
        print("StDev [pixel intensity]: %s" % stdev_pixel)
        print("Mean [cm^-1]: %s" % average_cmInv)
        print("StDev [cm^-1]: %s" % stdev_cmInv)
        print("Mean [HU]: %s" % average_HU)
        print("StDev [HU]: %s" % stdev_HU)
        print("Sphere radius [mm]: %s" % UI_RADIUS_mm)
        print("Number of points: %s" % len(roiPoints))

        # Save values
        allSphereAverage_pixel.append(average_pixel)
        allSphereStdev_pixel.append(stdev_pixel)
        allSphereAverage_cmInv.append(average_cmInv)
        allSphereStdev_cmInv.append(stdev_cmInv)
        allSphereAverage_HU.append(average_HU)
        allSphereStdev_HU.append(stdev_HU)

        # Update nrrd volume to include cylinders
        outVolume = iq.setValues(roiPoints, Volume, Volume.max()*5)

    # ---Uniformity Metric Calculations
    #tmpArray = np.array(allSphereAverage_HU)                                   #  CHOOSE UNITS [HU]
    tmpArray = np.array(allSphereAverage_cmInv)                                 #  CHOOSE UNITS [cm^-1]
    #tmpArray = np.array(allSphereAverage_pixel)                                #  CHOOSE UNITS [pixel]
    avgCenterSphere = tmpArray[0]
    avgPeripheralSphereArray = tmpArray[1:tmpArray.size]
    
    # Calculate Integral Non-Uniformity
    integralNonUniformity = ( tmpArray.max() - tmpArray.min() ) / ( tmpArray.max() + tmpArray.min() )
    
    # Calculate max Non-Uniformity Difference (also known as Uniformity)
    uniformityArray = abs(avgPeripheralSphereArray - avgCenterSphere)
    maxUniformity = uniformityArray.max()
    
    # Calculate Uniformity Index
    uniformityIndexArray = 100 * abs( (avgPeripheralSphereArray - avgCenterSphere) / avgCenterSphere )
    maxUniformityIndex = uniformityIndexArray.max()
   
    # Print output to screan
    print("\nIntegral Non-Uniformity [cm^-1/cm^-1]: %s" % np.around(integralNonUniformity,4))
    print("Difference ROI mean from Center mean [cm^-1]: %s" % np.around(uniformityArray,4))
    print("Difference Max [cm^-1]: %s" % np.around(maxUniformity,4))
    print("Uniformity Index [%%]: %s" % np.around(uniformityIndexArray,4))
    print("Uniformity Index Max [%%]: %s" % np.around(maxUniformityIndex,4))

    # Add ROIs to Volume
    if imageDim == 3:
        iq.saveOuputVolume(OUTPUT_FOLDER, outVolume, imageSpacings)
    elif imageDim == 2:
        iq.saveOuputVolume(OUTPUT_FOLDER, outVolume, [imageSpacings[0], imageSpacings[1]])
    
    iq.saveVolumeHdf5(OUTPUT_FOLDER, outVolume, imageSpacings,'OutputVolume')
    
def runNPS(filePath, dataset, xyzPoints):

    # Get data from dataset
    imageSpacings = dataset[HDF5_SPACINGS_KEY][:]
    imageDim = len(imageSpacings)
    scale = DEFAULT_IMAGE_SCALE
    tmpImageSizes = dataset[HDF5_DIMENSIONS_KEY][:]
    imageSizes = [int(i) for i in tmpImageSizes]    # convert list of floats to int

    # Assume group of reconstructions have same dimensions, spacings, and size
    if imageDim == 3:
        xN, yN, zN = imageSizes
        spacings = imageSpacings
    elif imageDim == 2:
        xN, yN = imageSizes
        zN = 1
        spacings = [imageSpacings[0], imageSpacings[1], 1]

    # Assign coordinates
    center = [int(xyzPoints[0]), int(xyzPoints[1]), int(xyzPoints[2])]    # Coordinate by pixel index
    print("ROI center index is %s" % center)

    # Generate list of files located in path
    fileList = iq.getHdf5FileList(filePath)

    # Load each file of interest
    numberOfVolumes = len(fileList)
    allVolumeArray_pixel = np.zeros((numberOfVolumes,xN,yN,zN))
    allVolumeArray_cmInv = np.zeros((numberOfVolumes,xN,yN,zN))
    allVolumeArray_HU = np.zeros((numberOfVolumes,xN,yN,zN))
    print("\n--Loading Volumes--")
    for m in range(0,numberOfVolumes):

        print(fileList[m])

        # Load file from list
        dataset = h5py.File(filePath + fileList[m] , 'r')

        # Get data from HDF5 file (backwards compatibility) TODO: pass HDF5 file not data
        tmpVolume_pixel = dataset[HDF5_VOLUME_KEY][:]
        tmpVolume_pixel = np.transpose(tmpVolume_pixel, HDF5_VOLUME_AXIS) # Transpose to correct coordinate frame

        # Convert Pixel Intensity to cm^-1 and HU
        tmpVolume_cmInv, tmpVolume_HU = iq.convert_Pixel2HU_Pixel2cmInv(scale, tmpVolume_pixel)

        # Put Volumes in array
        if imageDim == 3:
            allVolumeArray_pixel[m,:,:,:] = tmpVolume_pixel
            allVolumeArray_cmInv[m,:,:,:] = tmpVolume_cmInv    
            allVolumeArray_HU[m,:,:,:] = tmpVolume_HU
        elif imageDim == 2:
            allVolumeArray_pixel[m,:,:,0] = tmpVolume_pixel
            allVolumeArray_cmInv[m,:,:,0] = tmpVolume_cmInv    
            allVolumeArray_HU[m,:,:,0] = tmpVolume_HU

    #allVolumeOfInterest = allVolumeArray_HU                                 #  CHOOSE UNITS [HU]
    allVolumeOfInterest = allVolumeArray_cmInv                                 #  CHOOSE UNITS [cm^-1]
    print("--Loading Volumes complete--")

    # Define dimensions of NPS ROIs
    xWidth_pixel = NPS_ROI_DIM_X_mm / spacings[0]
    yWidth_pixel = NPS_ROI_DIM_Y_mm / spacings[1]
    if imageDim == 3:
        zDepth_pixel = NPS_ROI_DIM_Z_mm / spacings[2]
    elif imageDim == 2:
        zDepth_pixel = 0

    # Index value of ROI center point, should be the same for each Volume
    xCubeCent = center[0]
    yCubeCent = center[1]
    zCubeCent = center[2]
    
    # Find approximate range of indeces for cubic ROI
    xMaxDist = int(math.ceil(xWidth_pixel))
    yMaxDist = int(math.ceil(yWidth_pixel))
    zMaxDist = int(math.ceil(zDepth_pixel))

    xRange = range(xCubeCent-xMaxDist,xCubeCent+xMaxDist)
    yRange = range(yCubeCent-yMaxDist,yCubeCent+yMaxDist)
    if zDepth_pixel == 0:
        zRange = zCubeCent
        
        roiRange = [xRange, yRange, zRange]
        roiSize = [len(xRange), len(yRange), 1]         # This is needed because Python has trouble finding the size of a single number, e.g., zRange = 39
    else:
        zRange = range(zCubeCent-zMaxDist,zCubeCent+zMaxDist)    
        
        roiRange = [xRange, yRange, zRange]
        roiSize = [len(xRange), len(yRange), len(zRange)]

    # Obtain array for ROIs
    RoiValues = []      # data point intensity value
    RoiPoints = []      # data point location
    RoiArray = np.zeros((numberOfVolumes,roiSize[0],roiSize[1],roiSize[2]))
    if roiSize[2] == 1:
        for m in range(0,numberOfVolumes):
            for x in roiRange[0]:
                for y in roiRange[1]:
                    z = roiRange[2]
                    
                    # Shift new ROI volume indeces to start at 0
                    RoiArray[m,x-roiRange[0][0], y-roiRange[1][0], z-roiRange[2]] = allVolumeOfInterest[m,x,y,z]
                    
                    # Save only one ROI volume for outputVolume comparison and verification
                    if m == 0:
                        RoiPoints.append([x,y,z])
                        RoiValues.append(allVolumeOfInterest[m,x,y,z])
    else:
        for m in range(0,numberOfVolumes):
            for x in roiRange[0]:
                for y in roiRange[1]:
                    for z in roiRange[2]:
                        
                        # Shift new ROI volume indeces to start at 0
                        RoiArray[m,x-roiRange[0][0], y-roiRange[1][0], z-roiRange[2][0]] = allVolumeOfInterest[m,x,y,z]
                        
                        # Save only one ROI volume for outputVolume comparison and verification
                        if m == 0:
                            RoiPoints.append([x,y,z])
                            RoiValues.append(allVolumeOfInterest[m,x,y,z])

    #---+++++======= Calculate components of NPS for local region
    zeroPadAmnt = 2**NPS_ZPF    #for zero padding
    sumAbsSqrFftRoi = np.zeros((roiSize[0],roiSize[1]))
    sumAbsSqrFftRoi = np.pad(sumAbsSqrFftRoi,((0,zeroPadAmnt),(0,zeroPadAmnt)),mode='constant')      #for zero padding

    print RoiArray.shape
    expected = np.mean(RoiArray, 0)
    print expected.shape

    for m in range(0,numberOfVolumes):
        
        #-----ROI
        npsRoi = RoiArray[m,:,:,0]
        
        # Find x and y values of original ROI
        areaDim = npsRoi.shape
        xN = areaDim[0]
        yN = areaDim[1]
        xT = spacings[0]
        yT = spacings[1]
        x = np.arange(0,xN) * xT
        y = np.arange(0,yN) * yT
        
        ## Plot npsRoi
        #plt.figure('npsRoi_'+str(m))
        #plt.title('recon_'+str(m))
        #plt.imshow(npsRoi,interpolation='none',extent=[x.min(),x.max(),y.max(),y.min()])
        #plt.xlabel('y-axis distance [mm]')
        #plt.ylabel('x-axis distance [mm]')
        #plt.colorbar()      # display colorbar
        #plt.savefig(OUTPUT_FOLDER + 'npsRoi' + str(m))
        
        # Expected value (i.e. mean of ROI)      
        #expected = npsRoi.mean()
        
        #-----Difference from expected value
        deltaRoi = npsRoi - expected
        
        ## Plot deltaRoi
        #plt.figure('deltaRoi_' + str(m))
        #plt.title('recon_' + str(m))
        #plt.imshow(deltaRoi,interpolation='none',extent=[x.min(),x.max(),y.max(),y.min()])
        #plt.xlabel('y-axis distance [mm]')
        #plt.ylabel('x-axis distance [mm]')
        #plt.colorbar()      # display colorbar
        #plt.savefig(OUTPUT_FOLDER + 'deltaRoi' + str(m))
        
        #-----Zero pad the difference from average array
        paddedDeltaRoi = np.pad(deltaRoi, ((0,zeroPadAmnt),(0,zeroPadAmnt)), mode='constant')         #for zero padding
        
        # Find x and y values of padded ROI
        areaDim = paddedDeltaRoi.shape
        xPadN = areaDim[0]
        yPadN = areaDim[1]
        xPad = np.arange(0,xPadN) * xT
        yPad = np.arange(0,yPadN) * yT
        
        ## Plot paddedDeltaRoi
        #plt.figure('paddedDeltaRoi_' + str(m))
        #plt.title('recon_' + str(m))
        #plt.imshow(paddedDeltaRoi,interpolation='none',extent=[xPad.min(),xPad.max(),yPad.max(),yPad.min()])
        #plt.xlabel('y-axis distance [mm]')
        #plt.ylabel('x-axis distance [mm]')
        #plt.colorbar()      # display colorbar
        #plt.savefig(OUTPUT_FOLDER + 'paddedDeltaRoi' + str(m))
        
        #-----2D FFT of difference from average array
        fftRoi = np.fft.fft2(paddedDeltaRoi)                                    #for zero padding
        fftRoi = np.fft.fftshift(fftRoi)            #shift the zero-frequency component to the center of spectrum

        #-----Square the modulus
        absSqrFftRoi = np.abs(fftRoi)**2
        
        # Find xf and yf values in spatial frequency domain for FFT of ROI
        areaDim = absSqrFftRoi.shape
        xFftN = areaDim[0]
        yFftN = areaDim[1]
        xf = np.fft.fftfreq(xFftN,xT)
        yf = np.fft.fftfreq(yFftN,yT)
        idx = np.argsort(xf)
        idy = np.argsort(yf)
        
        ## Plot FFT of ROI
        #plt.figure('absSqrFftRoi' + str(m))
        #plt.title('recon_' + str(m))
        #plt.imshow(absSqrFftRoi,interpolation='none',extent=[xf.min(),xf.max(),yf.max(),yf.min()])
        #plt.xlabel('y-axis spatial frequency [mm^-1]')
        #plt.ylabel('x-axis spatial frequency [mm^-1]')
        #plt.colorbar()      # display colorbar
        #plt.savefig(OUTPUT_FOLDER + 'absSqrFftRoi' + str(m))
        
        # Sum together
        sumAbsSqrFftRoi = sumAbsSqrFftRoi + absSqrFftRoi

        # Print out various arrays for comparison
        np.savetxt((OUTPUT_FOLDER + 'npsRoi'+str(m)+'.out'), npsRoi)
        np.savetxt((OUTPUT_FOLDER + 'deltaRoi'+str(m)+'.out'), deltaRoi)
        np.savetxt((OUTPUT_FOLDER + 'paddedDeltaRoi'+str(m)+'.out'), paddedDeltaRoi)
        np.savetxt((OUTPUT_FOLDER + 'fftRoi'+str(m)+'.out'), fftRoi)
        np.savetxt((OUTPUT_FOLDER + 'absSqrFftRoi'+str(m)+'.out'), absSqrFftRoi)

    # ---Calculate NPS for specific region ---
    areaDim = sumAbsSqrFftRoi.shape    
    nps2DArray = (xT/xN) * (yT/yN) * (1.0/numberOfVolumes) * sumAbsSqrFftRoi
    np.savetxt((OUTPUT_FOLDER + 'nps2DArray.out'), nps2DArray)
    
    # Plot FFT of ROI
    plt.figure('nps2DArray')
    plt.title('2D NPS Map')
    plt.imshow(nps2DArray,interpolation='none',extent=[xf.min(),xf.max(),yf.max(),yf.min()])
    plt.xlabel('y-axis spatial frequency [mm^-1]')
    plt.ylabel('x-axis spatial frequency [mm^-1]')
    plt.colorbar()
    plt.savefig(OUTPUT_FOLDER + 'NPS_2D_Map')
    
 #--------Find 1D NPS
    print("\n--1D NPS--")
    
    # Find middle point of x and y direction of nps2DArray
    xMiddleInd = float(len(nps2DArray[:,0]))/2
    yMiddleInd = float(len(nps2DArray[0,:]))/2
    
    # Find middle line of x direction of nps2DArray  
    if yMiddleInd % 2 != 0:
        xMidLine = nps2DArray[:,int(yMiddleInd - .5)]   #if odd
    else:
        xMidHigh = nps2DArray[:,int(yMiddleInd)]          #if even
        xMidLow = nps2DArray[:,int(yMiddleInd-1)]
        xMidLine = (xMidHigh + xMidLow) / 2

    # Find middle of y direction of nps2DArray        
    if xMiddleInd % 2 != 0:
        yMidLine = nps2DArray[int(xMiddleInd - .5),:]   #if odd
    else:
        yMidHigh = nps2DArray[int(xMiddleInd),:]          #if even
        yMidLow = nps2DArray[int(xMiddleInd-1),:]
        yMidLine = (yMidHigh + yMidLow) / 2
    
    # Only positive half of line is needed
    xMidLineHalf = xMidLine[len(xMidLine)/2:]
    xfHalf = xf[:len(xMidLine)/2]
    xdf = np.diff(xfHalf).mean()
    yMidLineHalf = yMidLine[len(yMidLine)/2:]
    yfHalf = yf[:len(yMidLine)/2]
    ydf = np.diff(yfHalf).mean()

    # Compute the area under the 1D NPS
    xNoiseVariance = np.trapz(xMidLineHalf, dx=xdf)
    xNoiseMagnitude = np.sqrt(xNoiseVariance)
    yNoiseVariance = np.trapz(yMidLineHalf, dx=ydf)
    yNoiseMagnitude = np.sqrt(yNoiseVariance)
    
    # Find the Peak Frequency for 1D NPS
    xPeakFreq = xfHalf[np.argmax(xMidLineHalf)]
    yPeakFreq = yfHalf[np.argmax(yMidLineHalf)]
    
    # Display results
    print("\nNoise Variance of NPSx: {:.6f}".format(xNoiseVariance))
    print("Noise Magnitude of NPSx: {:.6f}".format(xNoiseMagnitude))
    print("Peak Frequency of NPSx [mm^-1]: {:.6f}".format(xPeakFreq))
    print("\nNoise Variance of NPSy: {:.6f}".format(yNoiseVariance))
    print("Noise Magnitude of NPSy: {:.6f}".format(yNoiseMagnitude))
    print("Peak Frequency of NPSy [mm^-1]: {:.6f}".format(yPeakFreq))
    
    # Plot 1D NPS
    plt.figure('1D NPS', figsize=(13,8))
    plt.subplot(122)
    plt.imshow(nps2DArray, extent=(xf.min(), xf.max(), yf.max(), yf.min()), interpolation='nearest')
    plt.ylabel('fx [mm^-1]', fontsize=16)
    plt.xlabel('fy [mm^-1]', fontsize=16)
    plt.colorbar()      # display colorbar
    plt.plot([0,0],[xf.max(),0],'ro-')
    plt.plot([0,yf.max()],[0,0],'go-')
    
    plt.subplot(221)
    plt.plot(xfHalf, xMidLineHalf, 'ro-')
    plt.ylabel('NPSx [(cm^-1)^2 mm^2]', fontsize=16)
    plt.xlabel('Spatial Frequency fx [mm^-1]', fontsize=16)
    
    plt.subplot(223)
    plt.plot(yfHalf, yMidLineHalf, 'go-')
    plt.ylabel('NPSy [(cm^-1)^2 mm^2]', fontsize=16)
    plt.xlabel('Spatial Frequency fy [mm^-1]', fontsize=16)
    plt.savefig(OUTPUT_FOLDER + 'NPS_1Dx_1Dy_2DMap_Plot')
    
 #--------Find 2D NPS
    print("\n--2D NPS--")
    
    # Find spatial frequency domain distances
    distFromCenter = []
    npsValueFromCenter = []    
    for xInd in range(len(xf)):
        for yInd in range(len(yf)):
            xDiff = (xInd - xMiddleInd) * xdf
            yDiff = (yInd - yMiddleInd) * ydf
            tmpDist = np.sqrt( xDiff**2 + yDiff**2 )
            distFromCenter.append(tmpDist)
            npsValueFromCenter.append(nps2DArray[xInd,yInd])

    # Save distances and values
    np.savetxt((OUTPUT_FOLDER + 'distFromCenter.out'), distFromCenter)
    np.savetxt((OUTPUT_FOLDER + 'npsValueFromCenter.out'), npsValueFromCenter)
    
    # Convert lists to arrays
    distFromCenter = np.array(distFromCenter)
    npsValueFromCenter = np.array(npsValueFromCenter)

    # Radially bin values based on distances from center
    binSize = xdf / NPS_BIN_SIZE_DIVISOR
    npsBinList = []
    for index in range(0,1+int(math.ceil(distFromCenter.max()/(binSize)))):
        minRadius = binSize * index
        maxRadius = binSize * (index + 1)
        valuesWithinBin = npsValueFromCenter[ (minRadius <= distFromCenter) & (distFromCenter < maxRadius) ]
        
        # Verify values are contained in bin
        if len(valuesWithinBin) == 0:
            print("\nBinning stopped - empty bin found.")
            print("Total number of bins {}".format(index-1))
            break
        
        npsBinList.append(valuesWithinBin.mean())
        
        # Values within range of Nyquist Frequency
        if maxRadius >= xf.max():
            print("\nBinning stopped - Nyquist Frequency reached.")
            print("Total number of bins {}".format(index))
            break
    npsBinArray = np.array(npsBinList)
    
    # Save nps binned array
    np.savetxt((OUTPUT_FOLDER + 'npsBinArray.out'), npsBinArray)
    
    # X-axis for radial bins in spatial frequency domain
    rN = len(npsBinArray)
    rf = np.array(range(0,rN,1)) * binSize
    np.savetxt((OUTPUT_FOLDER + 'rf.out'), rf)
    
    # Compute the area under the 2D NPS
    rNoiseVariance = np.trapz(npsBinArray, dx=binSize)
    rNoiseMagnitude = np.sqrt(rNoiseVariance)
    
    # Find the Peak Frequency for 2D NPS
    rPeakFreq = rf[np.argmax(npsBinArray)]
    
    # Display results
    print("\nNoise Variance of 2D NPS: {:.6f}".format(rNoiseVariance))
    print("Noise Magnitude of 2D NPS: {:.6f}".format(rNoiseMagnitude))
    print("Peak Frequency of 2D NPS [mm^-1]: {:.6f}".format(rPeakFreq))

    # Plot NPS 
    plt.figure('1D NPS from radial binning of 2D Map', figsize=(9,8))
    plt.plot(rf,npsBinArray, linestyle='-', marker='o')
    plt.ylabel('NPS [(cm^-1)^2 mm^2]', fontsize=16)
    plt.xlabel('Spatial Frequency fr [mm^-1]', fontsize=16)
    plt.savefig(OUTPUT_FOLDER + 'NPS_1D_Plot')

    ## Add ROIs to Volume
    outVolume = iq.setValues(RoiPoints, allVolumeArray_pixel[0,:,:,:], MAX_PIXEL_VALUE)     # only initial volume for verification

    iq.saveOuputVolume(OUTPUT_FOLDER, outVolume, spacings)
    
    iq.saveVolumeHdf5(OUTPUT_FOLDER, outVolume, imageSpacings,'OutputVolume')
    
    #plt.show()
    
def runNPS3D(filePath, dataset, xyzPoints):

    # Get data from dataset
    imageSpacings = dataset[HDF5_SPACINGS_KEY][:]
    imageDim = len(imageSpacings)
    scale = DEFAULT_IMAGE_SCALE
    tmpImageSizes = dataset[HDF5_DIMENSIONS_KEY][:]
    imageSizes = [int(i) for i in tmpImageSizes]    # convert list of floats to int

    
    # Assume group of reconstructions have same spacings, and size
    xN, yN, zN = imageSizes
    spacings = imageSpacings

    # Assign coordinates
    center = [int(xyzPoints[0]), int(xyzPoints[1]), int(xyzPoints[2])]    # Coordinate by pixel index
    print("ROI center index is %s" % center)
    
    # Generate list of files located in path
    fileList = iq.getHdf5FileList(filePath)

    # Load each file of interest
    numberOfVolumes = len(fileList)
    allVolumeArray_pixel = np.zeros((numberOfVolumes,xN,yN,zN))
    allVolumeArray_cmInv = np.zeros((numberOfVolumes,xN,yN,zN))
    allVolumeArray_HU = np.zeros((numberOfVolumes,xN,yN,zN))
    print("\n--Loading Volumes--")
    for m in range(0,numberOfVolumes):
        
        print(fileList[m])
       
        # Load file from list
        dataset = h5py.File(filePath + fileList[m] , 'r')

        # Get data from HDF5 file (backwards compatibility) TODO: pass HDF5 file not data
        tmpVolume_pixel = dataset[HDF5_VOLUME_KEY][:]
        tmpVolume_pixel = np.transpose(tmpVolume_pixel, HDF5_VOLUME_AXIS) # Transpose to correct coordinate frame

        # Convert Pixel Intensity to cm^-1 and HU
        tmpVolume_cmInv, tmpVolume_HU = iq.convert_Pixel2HU_Pixel2cmInv(scale, tmpVolume_pixel)

        # Put Volumes in array
        allVolumeArray_pixel[m,:,:,:] = tmpVolume_pixel
        allVolumeArray_cmInv[m,:,:,:] = tmpVolume_cmInv    
        allVolumeArray_HU[m,:,:,:] = tmpVolume_HU
    
    allVolumeOfInterest = allVolumeArray_cmInv                          #  CHOOSE UNITS [cm^-1]
    print("--Loading Volumes complete--")

    # Define dimensions of NPS ROIs
    xWidth_pixel = NPS_ROI_DIM_X_mm / spacings[0]
    yWidth_pixel = NPS_ROI_DIM_Y_mm / spacings[1]
    zDepth_pixel = NPS_ROI_DIM_Z_mm / spacings[2]
    
    # Index value of ROI center point, should be the same for each Volume
    xCubeCent = center[0]
    yCubeCent = center[1]
    zCubeCent = center[2]
    
    # Find approximate range of indeces for cubic ROI
    xMaxDist = int(math.ceil(xWidth_pixel))
    yMaxDist = int(math.ceil(yWidth_pixel))
    zMaxDist = int(math.ceil(zDepth_pixel))

    xRange = range(xCubeCent-xMaxDist,xCubeCent+xMaxDist)
    xLength = len(xRange)
    if yWidth_pixel == 0:
        yRange = [yCubeCent]
        yLength = 1
    else:
        yRange = range(yCubeCent-yMaxDist,yCubeCent+yMaxDist)
        yLength = len(yRange)
                
    if zDepth_pixel == 0:
        zRange = [zCubeCent]
        zLength = 1
    else:
        zRange = range(zCubeCent-zMaxDist,zCubeCent+zMaxDist)    
        zLength = len(zRange)

    roiRange = [xRange, yRange, zRange]
    roiSize = [xLength, yLength, zLength]         # This is needed because Python has trouble finding the size of a single number, e.g., zRange = 39

    # Obtain array for ROIs and corresponding values
    RoiValues = []      # data point intensity value
    RoiPoints = []      # data point location
    RoiArray = np.zeros((numberOfVolumes,xLength,yLength,zLength))
    for m in range(0,numberOfVolumes):
        for x in xRange:
            for y in yRange:
                for z in zRange:
                    
                    # Shift new ROI volume indeces to start at 0
                    RoiArray[m,x-xRange[0], y-yRange[0], z-zRange[0]] = allVolumeOfInterest[m,x,y,z]
                    
                    # Save only one ROI volume for outputVolume comparison and verification
                    if m == 0:
                        RoiPoints.append([x,y,z])
                        RoiValues.append(allVolumeOfInterest[m,x,y,z])

        plt.imshow(RoiArray[m,:,:,0])
        plt.savefig(OUTPUT_FOLDER + 'image' + str(m) + '.png')
        #plt.show()

    #---+++++======= Calculate components of NPS for local region
    zeroPadAmnt = 2**NPS_ZPF    #for zero padding
    #sumAbsSqrFftRoi = np.zeros((roiSize[0],roiSize[2]))
    sumAbsSqrFftRoi = np.zeros((roiSize[0],roiSize[1],roiSize[2]))
    print('\n')
    print(sumAbsSqrFftRoi.shape)
    sumAbsSqrFftRoi = np.pad(sumAbsSqrFftRoi,((0,zeroPadAmnt),(0,zeroPadAmnt),(0,zeroPadAmnt)),mode='constant')      #for zero padding
    sumAbsSqrShiftFftRoi = sumAbsSqrFftRoi

    expected = np.mean(RoiArray, 0)
    expected = expected - expected.mean()
    plt.imshow(RoiArray[m,:,:,0])
    plt.savefig(OUTPUT_FOLDER + 'expected' + '.png')
    #plt.show()

    for m in range(0,numberOfVolumes):
        
        #-----ROI
        npsRoi = RoiArray[m,:,:,:]
        npsRoi = npsRoi - npsRoi.mean()
        
        # Find x and y values of original ROI
        areaDim = npsRoi.shape
        xN = areaDim[0]
        yN = areaDim[1]
        zN = areaDim[2]
        xT = spacings[0]
        yT = spacings[1]
        zT = spacings[2]
        x = np.arange(0,xN) * xT
        y = np.arange(0,yN) * yT
        z = np.arange(0,zN) * zT
        
        #iq.saveVolume(OUTPUT_FOLDER, npsRoi, [xT, yT, zT], 'npsRoi_'+str(m))
        
        # Expected value (i.e. mean of ROI)      
        #expected = npsRoi.mean()
        
        #-----Difference from expected value
        deltaRoi = npsRoi - expected
        plt.imshow(deltaRoi[:,:,0])
        plt.savefig(OUTPUT_FOLDER + 'subtracted' + str(m) + '.png')
        #plt.show()

        #iq.saveVolume(OUTPUT_FOLDER, npsRoi, [xT, yT, zT], 'deltaRoi' + str(m))
        
        #-----Zero pad the difference from average array
        paddedDeltaRoi = np.pad(deltaRoi, ((0,zeroPadAmnt),(0,zeroPadAmnt),(0,zeroPadAmnt)), mode='constant')
        
        # Find x and y values of padded ROI
        areaDim = paddedDeltaRoi.shape
        xPadN = areaDim[0]
        yPadN = areaDim[1]
        zPadN = areaDim[2]
        xPad = np.arange(0,xPadN) * xT
        yPad = np.arange(0,yPadN) * yT
        zPad = np.arange(0,zPadN) * zT

        #iq.saveVolume(OUTPUT_FOLDER, npsRoi, [xT, yT, zT], 'paddedDeltaRoi' + str(m))
        
        #-----3D FFT of difference from average array
        fftRoi = np.fft.fftn(paddedDeltaRoi)                                    #for zero padding
        fftShiftRoi = np.fft.fftshift(fftRoi)            #shift the zero-frequency component to the center of spectrum

        # Find xf and yf values in spatial frequency domain for FFT of ROI
        areaDim = fftRoi.shape
        xFftN = areaDim[0]
        yFftN = areaDim[1]
        zFftN = areaDim[2]
        xf = np.fft.fftfreq(xFftN,xT)
        yf = np.fft.fftfreq(yFftN,yT)
        zf = np.fft.fftfreq(zFftN,zT)
        idx = np.argsort(xf)
        idy = np.argsort(yf)
        idz = np.argsort(zf)
        xdf = np.diff(xf[:10]).mean()
        ydf = np.diff(yf[:10]).mean()
        zdf = np.diff(zf[:10]).mean()

        #-----Square the modulus
        absSqrFftRoi = np.abs(fftRoi)**2
        absSqrShiftFftRoi = np.abs(fftShiftRoi)**2
        
        #iq.saveVolume(OUTPUT_FOLDER, npsRoi, [xT, yT, zT], 'absSqrShiftFftRoi' + str(m))
        
        # Sum together
        sumAbsSqrFftRoi = sumAbsSqrFftRoi + absSqrFftRoi
        sumAbsSqrShiftFftRoi = sumAbsSqrShiftFftRoi + absSqrShiftFftRoi
        
    #iq.saveVolume(OUTPUT_FOLDER, npsRoi, [xT, yT, zT], 'sumAbsSqrShiftFftRoi' + str(m))

    # ---Calculate NPS for specific region ---
    areaDim = sumAbsSqrShiftFftRoi.shape
    nps3DArray = (xT/xN)* (yT/yN) * (zT/zN) * (1.0/numberOfVolumes) * sumAbsSqrShiftFftRoi
  
    iq.saveVolume(OUTPUT_FOLDER, nps3DArray, [xdf, ydf, zdf], 'nps3DVolume')
    
 #--------Find 1D NPS
    print("\n--1D NPS--")
    
    # Find middle point of x and y direction of nps3DArray
    xMiddleInd = float(len(nps3DArray[:,0,0]))/2
    yMiddleInd = float(len(nps3DArray[0,:,0]))/2
    zMiddleInd = float(len(nps3DArray[0,0,:]))/2
    
    print(int(xMiddleInd))
    print(int(yMiddleInd))
    print(int(zMiddleInd))
    
    #-----X-Y PLANE
    print('\n-----x-y plane-----')
    
    # Create x-y plane at fz = 0
    xyNps2dArray = nps3DArray[:,:,int(zMiddleInd)]
    np.savetxt((OUTPUT_FOLDER + 'xyNps2dArray.out'), xyNps2dArray)
    
    # Find values along fx when fy = 0 
    xMidLine = xyNps2dArray[:,int(yMiddleInd)]

    # Find values along fy when fx = 0
    yMidLine = xyNps2dArray[int(xMiddleInd),:]
    
    # Only positive half of line is needed
    xMidLineHalf = xMidLine[len(xMidLine)/2:]
    xfHalf = xf[:len(xMidLine)/2]
    xdf = np.diff(xfHalf).mean()
    yMidLineHalf = yMidLine[len(yMidLine)/2:]
    yfHalf = yf[:len(yMidLine)/2]
    ydf = np.diff(yfHalf).mean()

    # Compute the area under the 1D NPS
    xNoiseVariance = np.trapz(xMidLineHalf, dx=xdf)
    xNoiseMagnitude = np.sqrt(xNoiseVariance)
    yNoiseVariance = np.trapz(yMidLineHalf, dx=ydf)
    yNoiseMagnitude = np.sqrt(yNoiseVariance)
    
    # Find the Peak Frequency for 1D NPS
    xPeakFreq = xfHalf[np.argmax(xMidLineHalf)]
    yPeakFreq = yfHalf[np.argmax(yMidLineHalf)]
    
    # Display results
    print("\nNoise Variance of NPSx: {:.4E}".format(xNoiseVariance))
    print("Noise Magnitude of NPSx: {:.4E}".format(xNoiseMagnitude))
    print("Peak Frequency of NPSx [mm^-1]: {:.4E}".format(xPeakFreq))
    print("\nNoise Variance of NPSy: {:.4E}".format(yNoiseVariance))
    print("Noise Magnitude of NPSy: {:.4E}".format(yNoiseMagnitude))
    print("Peak Frequency of NPSy [mm^-1]: {:.4E}".format(yPeakFreq))
    
    # Plot 1D NPS
    plt.figure('1D NPS: x-y plane', figsize=(13,8))
    plt.subplot(122)
    plt.imshow(xyNps2dArray, extent=(yf.min(),yf.max(), xf.max(),xf.min()), interpolation='none',origin='upper')
    plt.ylabel('fx [mm^-1]', fontsize=16)
    plt.xlabel('fy [mm^-1]', fontsize=16)
    plt.colorbar(format='%.02e')
    plt.plot([0,0],[0,xf.max()],'ro-')
    plt.plot([0,yf.max()],[0,0],'go-')
    
    plt.subplot(221)
    plt.plot(xfHalf, xMidLineHalf, 'ro-')
    plt.ylabel('NPSx [(cm^-1)^2 mm^3]', fontsize=16)
    plt.xlabel('Spatial Frequency fx [mm^-1]', fontsize=16)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.subplot(223)
    plt.plot(yfHalf, yMidLineHalf, 'go-')
    plt.ylabel('NPSy [(cm^-1)^2 mm^3]', fontsize=16)
    plt.xlabel('Spatial Frequency fy [mm^-1]', fontsize=16)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.savefig(OUTPUT_FOLDER + 'NPS_1Dx_1Dy_2DMap_Plot')
    
    #-----X-Z PLANE
    print('\n-----x-z plane-----')
    
    # Create x-z plane at fy = 0
    xzNps2dArray = nps3DArray[:,int(yMiddleInd),:]
    np.savetxt((OUTPUT_FOLDER + 'xzNps2dArray.out'), xzNps2dArray)
        
    # Find values along fx when fz = 0
    xMidLine = xzNps2dArray[:,int(zMiddleInd)]

    # Find values along fz when fx = 0
    zMidLine = xzNps2dArray[int(xMiddleInd),:]
    
    # Only positive half of line is needed
    xMidLineHalf = xMidLine[len(xMidLine)/2:]
    xfHalf = xf[:len(xMidLine)/2]
    xdf = np.diff(xfHalf).mean()
    zMidLineHalf = zMidLine[len(zMidLine)/2:]
    zfHalf = zf[:len(zMidLine)/2]
    zdf = np.diff(zfHalf).mean()

    # Compute the area under the 1D NPS
    xNoiseVariance = np.trapz(xMidLineHalf, dx=xdf)
    xNoiseMagnitude = np.sqrt(xNoiseVariance)
    zNoiseVariance = np.trapz(zMidLineHalf, dx=zdf)
    zNoiseMagnitude = np.sqrt(zNoiseVariance)
    
    # Find the Peak Frequency for 1D NPS
    xPeakFreq = xfHalf[np.argmax(xMidLineHalf)]
    zPeakFreq = zfHalf[np.argmax(zMidLineHalf)]
    
    # Display results
    print("\nNoise Variance of NPSx: {:.4E}".format(xNoiseVariance))
    print("Noise Magnitude of NPSx: {:.4E}".format(xNoiseMagnitude))
    print("Peak Frequency of NPSx [mm^-1]: {:.4E}".format(xPeakFreq))
    print("\nNoise Variance of NPSz: {:.4E}".format(zNoiseVariance))
    print("Noise Magnitude of NPSz: {:.4E}".format(zNoiseMagnitude))
    print("Peak Frequency of NPSz [mm^-1]: {:.4E}".format(zPeakFreq))
    
    # Plot 1D NPS
    plt.figure('1D NPS: x-z plane', figsize=(13,8))
    plt.subplot(122)
    plt.imshow(xzNps2dArray, extent=(zf.min(), zf.max(), xf.max(), xf.min()), interpolation='none')
    plt.ylabel('fx [mm^-1]', fontsize=16)
    plt.xlabel('fz [mm^-1]', fontsize=16)
    plt.colorbar(format='%.02e')
    plt.plot([0,0],[0,xf.max()],'ro-')
    plt.plot([0,zf.max()],[0,0],'go-')
    
    plt.subplot(221)
    plt.plot(xfHalf, xMidLineHalf, 'ro-')
    plt.ylabel('NPSx [(cm^-1)^2 mm^3]', fontsize=16)
    plt.xlabel('Spatial Frequency fx [mm^-1]', fontsize=16)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.subplot(223)
    plt.plot(zfHalf, zMidLineHalf, 'go-')
    plt.ylabel('NPSz [(cm^-1)^2 mm^3]', fontsize=16)
    plt.xlabel('Spatial Frequency fz [mm^-1]', fontsize=16)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.savefig(OUTPUT_FOLDER + 'NPS_1Dx_1Dz_2DMap_Plot')
    
    #-----Y-Z PLANE
    print('\n-----y-z plane-----')
    
    # Create y-z plane at fx = 0
    yzNps2dArray = nps3DArray[int(xMiddleInd),:,:]
    np.savetxt((OUTPUT_FOLDER + 'yzNps2dArray.out'), yzNps2dArray)

    # Find values along fy when fz = 0
    yMidHigh = yzNps2dArray[:,int(zMiddleInd)]

    # Find values along fz when fy = 0
    zMidHigh = yzNps2dArray[int(yMiddleInd),:]
    
    # Only positive half of line is needed
    yMidLineHalf = yMidLine[len(yMidLine)/2:]
    yfHalf = yf[:len(yMidLine)/2]
    ydf = np.diff(yfHalf).mean()
    zMidLineHalf = zMidLine[len(zMidLine)/2:]
    zfHalf = zf[:len(zMidLine)/2]
    zdf = np.diff(zfHalf).mean()

    # Compute the area under the 1D NPS
    yNoiseVariance = np.trapz(yMidLineHalf, dx=ydf)
    yNoiseMagnitude = np.sqrt(yNoiseVariance)
    zNoiseVariance = np.trapz(zMidLineHalf, dx=zdf)
    zNoiseMagnitude = np.sqrt(zNoiseVariance)
    
    # Find the Peak Frequency for 1D NPS
    yPeakFreq = yfHalf[np.argmax(yMidLineHalf)]
    zPeakFreq = zfHalf[np.argmax(zMidLineHalf)]
    
    # Display results
    print("\nNoise Variance of NPSy: {:.4E}".format(yNoiseVariance))
    print("Noise Magnitude of NPSy: {:.4E}".format(yNoiseMagnitude))
    print("Peak Frequency of NPSy [mm^-1]: {:.4E}".format(yPeakFreq))
    print("\nNoise Variance of NPSz: {:.4E}".format(zNoiseVariance))
    print("Noise Magnitude of NPSz: {:.4E}".format(zNoiseMagnitude))
    print("Peak Frequency of NPSz [mm^-1]: {:.4E}".format(zPeakFreq))

    # Plot 1D NPS
    plt.figure('1D NPS: y-z plane', figsize=(13,8))
    plt.subplot(122)
    plt.imshow(yzNps2dArray, extent=(zf.min(), zf.max(), yf.max(), yf.min()), interpolation='none')
    plt.ylabel('fy [mm^-1]', fontsize=16)
    plt.xlabel('fz [mm^-1]', fontsize=16)
    plt.colorbar(format='%.02e')
    plt.plot([0,0],[0,yf.max()],'ro-')
    plt.plot([0,zf.max()],[0,0],'go-')
    
    plt.subplot(221)
    plt.plot(yfHalf, yMidLineHalf, 'ro-')
    plt.ylabel('NPSy [(cm^-1)^2 mm^3]', fontsize=16)
    plt.xlabel('Spatial Frequency fy [mm^-1]', fontsize=16)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.subplot(223)
    plt.plot(zfHalf, zMidLineHalf, 'go-')
    plt.ylabel('NPSz [(cm^-1)^2 mm^3]', fontsize=16)
    plt.xlabel('Spatial Frequency fz [mm^-1]', fontsize=16)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.savefig(OUTPUT_FOLDER + 'NPS_1Dy_1Dz_2DMap_Plot')
    
 #--------Find 2D NPS
    print("\n--2D NPS--")
    
    # Find spatial frequency domain distances
    distFromCenter = []
    npsValueFromCenter = []    
    for xInd in range(len(xf)):
        for yInd in range(len(yf)):
            xDiff = (xInd - xMiddleInd) * xdf
            yDiff = (yInd - yMiddleInd) * ydf
            tmpDist = np.sqrt( xDiff**2 + yDiff**2 )
            distFromCenter.append(tmpDist)
            npsValueFromCenter.append(xyNps2dArray[xInd,yInd])

    # Save distances and values
    np.savetxt((OUTPUT_FOLDER + 'distFromCenter.out'), distFromCenter)
    np.savetxt((OUTPUT_FOLDER + 'npsValueFromCenter.out'), npsValueFromCenter)
    
    # Convert lists to arrays
    distFromCenter = np.array(distFromCenter)
    npsValueFromCenter = np.array(npsValueFromCenter)

    # Radially bin values based on distances from center
    binSize = xdf / NPS_BIN_SIZE_DIVISOR
    npsBinList = []
    for index in range(0,1+int(math.ceil(distFromCenter.max()/(binSize)))):
        minRadius = binSize * index
        maxRadius = binSize * (index + 1)
        valuesWithinBin = npsValueFromCenter[ (minRadius <= distFromCenter) & (distFromCenter < maxRadius) ]
        
        # Verify values are contained in bin
        if len(valuesWithinBin) == 0:
            print("\nBinning stopped - empty bin found.")
            print("Total number of bins {}".format(index-1))
            break
        
        npsBinList.append(valuesWithinBin.mean())
        
        # Values within range of Nyquist Frequency
        if maxRadius >= xf.max():
            print("\nBinning stopped - Nyquist Frequency reached.")
            print("Total number of bins {}".format(index))
            break
    npsBinArray = np.array(npsBinList)

    # Save nps binned array
    np.savetxt((OUTPUT_FOLDER + 'npsBinArray.out'), npsBinArray)
    
    # X-axis for radial bins in spatial frequency domain
    rN = len(npsBinArray)
    rf = np.array(range(0,rN,1)) * binSize
    np.savetxt((OUTPUT_FOLDER + 'rf.out'), rf)
    
    # Compute the area under the 2D NPS
    rNoiseVariance = np.trapz(npsBinArray, dx=binSize)
    rNoiseMagnitude = np.sqrt(rNoiseVariance)
    
    # Find the Peak Frequency for 2D NPS
    rPeakFreq = rf[np.argmax(npsBinArray)]
    
    # Display results
    print("\nNoise Variance of 2D NPS: {:.4E}".format(rNoiseVariance))
    print("Noise Magnitude of 2D NPS: {:.4E}".format(rNoiseMagnitude))
    print("Peak Frequency of 2D NPS [mm^-1]: {:.4E}".format(rPeakFreq))

    # Plot NPS map in fx and fy
    plt.figure('NPS Map fx-fy', figsize=(10,8))
    plt.imshow(xyNps2dArray, extent=(yf.min(),yf.max(), xf.max(),xf.min()), interpolation='none',origin='upper')
    plt.ylabel('fx [mm^-1]', fontsize=16)
    plt.xlabel('fy [mm^-1]', fontsize=16)
    plt.colorbar(format='%.02e')
    plt.plot([0,0],[0,xf.max()],'ro-')
    plt.plot([0,yf.max()],[0,0],'go-')
    plt.savefig(OUTPUT_FOLDER + 'NPS Map fx-fy.png')

    # Plot NPS radial in X and Y
    plt.figure('NPS Plot fr', figsize=(10,8))
    plt.plot(rf,npsBinArray, linestyle='-', marker='o')
    plt.ylabel('NPS [(cm^-1)^2 mm^3]', fontsize=16)
    plt.xlabel('Spatial Frequency fr [mm^-1]', fontsize=16)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(OUTPUT_FOLDER + 'NPS Plot fr.png')
 
    # Plot NPS map in fx and fz
    plt.figure('NPS Map fx-fz', figsize=(10,8))
    plt.imshow(xzNps2dArray, extent=(zf.min(), zf.max(), xf.max(), xf.min()), interpolation='none')
    plt.ylabel('fx [mm^-1]', fontsize=16)
    plt.xlabel('fz [mm^-1]', fontsize=16)
    plt.colorbar(format='%.02e')
    plt.plot([0,0],[0,xf.max()],'ro-')
    plt.plot([0,zf.max()],[0,0],'go-')
    plt.savefig(OUTPUT_FOLDER + 'NPS Map fx-fz.png')

    ## Add ROIs to Volume
    print('\nSaving Output Volume')
    outVolume = iq.setValues(RoiPoints, allVolumeArray_pixel[0,:,:,:], MAX_PIXEL_VALUE)     # only initial volume for verification
    
    iq.saveOuputVolume(OUTPUT_FOLDER, outVolume, spacings)
    
    iq.saveVolumeHdf5(OUTPUT_FOLDER, outVolume, imageSpacings,'OutputVolume')
    
    #plt.show()

def runZST(filePath, dataset, xyzPoints):

    # Get data from dataset
    imageSpacings = dataset[HDF5_SPACINGS_KEY][:]
    scale = DEFAULT_IMAGE_SCALE
    tmpImageSizes = dataset[HDF5_DIMENSIONS_KEY][:]
    imageSizes = [int(i) for i in tmpImageSizes]    # convert list of floats to int

    # Assign coordinates
    center = [int(xyzPoints[0]), int(xyzPoints[1]), int(xyzPoints[2])]

    # Get spacings and image size
    xN, yN, zN = imageSizes
    xSpc, ySpc, zSpc = imageSpacings
    spacings = imageSpacings

    # Print center points to screen
    print("Center point is: %s" % center)
    
    # Generate list nrrd of files located in path
    fileList = iq.getHdf5FileList(filePath)
    
    # Use spacing to determine x-axis based on step size of grid shift
    numberOfVolumes = len(fileList)
    stepSize = round(spacings[2]/ZST_GRID_DIVISOR,2)
    print
    print('z-spacing: {}'.format(spacings[2]))
    print('z-grid step size {}'.format(stepSize))    

    # Load each file of interest (ZST_GRID_SHIFT_NUM is the number of files)
    allVolumeArray_pixel = np.zeros((numberOfVolumes,xN,yN,zN))
    allVolumeArray_cmInv = np.zeros((numberOfVolumes,xN,yN,zN))
    allVolumeArray_HU = np.zeros((numberOfVolumes,xN,yN,zN))
    print("\n--Loading Volumes--")
    for m in range(0,numberOfVolumes):
        
        print(fileList[m])
        
        # Load file from list
        dataset = h5py.File(filePath + fileList[m] , 'r')

        # Get data from HDF5 file (backwards compatibility) TODO: pass HDF5 file not data
        tmpVolume_pixel = dataset[HDF5_VOLUME_KEY][:]
        tmpVolume_pixel = np.transpose(tmpVolume_pixel, HDF5_VOLUME_AXIS) # Transpose to correct coordinate frame

        # Convert Pixel Intensity to cm^-1 and HU
        tmpVolume_cmInv, tmpVolume_HU = iq.convert_Pixel2HU_Pixel2cmInv(scale, tmpVolume_pixel)

        # Put Volumes in array
        allVolumeArray_pixel[m,:,:,:] = tmpVolume_pixel
        allVolumeArray_cmInv[m,:,:,:] = tmpVolume_cmInv    
        allVolumeArray_HU[m,:,:,:] = tmpVolume_HU
    
    print("--Loading Volumes complete--")
    
    # choose units
    allVolumeOfInterest = allVolumeArray_cmInv
    
    # Save array of all volumes and their data
    np.save((OUTPUT_FOLDER + 'allLoadedVolumesArray'), allVolumeOfInterest)  # save the file as "allLoadedVolumesArray.npy" 

    ## Load array
    #allVolumeOfInterest = np.load((OUTPUT_FOLDER + 'allLoadedVolumesArray.npy')) # loads your saved array

    # Get range of pixel coordinates surrounding the pixel of interest (ZST_PIXEL_SPREAD is the number of pixels)
    zRangeFlipped = np.flipud(range(center[2]-ZST_PIXEL_SPREAD/2,center[2]+ZST_PIXEL_SPREAD/2 + 1))
    
    # Get each pixel coordinates' (zRangeFlipped) set of values for each grid shift (numberOfVolumes)
    intensityValueArray = np.zeros((ZST_PIXEL_SPREAD+1,ZST_GRID_SHIFT_NUM+1))
    for m in range(0,len(zRangeFlipped)):
        for n in range(0,numberOfVolumes):
        
            # Get pixel value in volume n
            intensityValueArray[m,n] = allVolumeOfInterest[n,center[0],center[1],zRangeFlipped[m]]

    loi = intensityValueArray[:,::-1].flatten();

    # Get x-axis values in space domain
    z_mm = np.arange(0,len(loi)) * stepSize
 
    # Remove offset
    loiOffsetRemoved = loi - loi.min()

    # Get points for FWHM
    # ---Separate the rising and falling halves of line profile
    loiRise = loiOffsetRemoved[:np.argmax(loiOffsetRemoved)+1]
    loiFall = loiOffsetRemoved[np.argmax(loiOffsetRemoved):]
    
    # Get Max Value
    yMax = loiOffsetRemoved.max()
    
    # ---Find x0, where 1/2 max occurs on rise of line profile
    percent = 50.0/100
    ind_xA = np.argmax(loiRise[loiRise < percent*yMax])
    ind_xB = ind_xA + 1
    
    slope0 = (loiRise[ind_xB] - loiRise[ind_xA])/(z_mm[ind_xB] - z_mm[ind_xA])
    intercept0 = loiRise[ind_xA] - (slope0 * z_mm[ind_xA])
    x0 = (percent*yMax - intercept0) / slope0
    
    # ---Find x1, where 1/2 max occurs on fall of line profile
    ind_xD = np.argmax([loiFall < percent*yMax]) + np.argmax(loiOffsetRemoved)
    ind_xC = ind_xD - 1
    slope1 = (loiOffsetRemoved[ind_xD] - loiOffsetRemoved[ind_xC])/(z_mm[ind_xD] - z_mm[ind_xC])
    intercept1 = loiOffsetRemoved[ind_xC] - (slope1 * z_mm[ind_xC])
    x1 = ( (percent*yMax - intercept1) / slope1 )

    # Plot line profile
    plt.plot(x0,percent*yMax,'ro-', label='x0', markersize=10)
    plt.plot(x1,percent*yMax,'go-', label='x1', markersize=10)
    plt.plot(z_mm,loiOffsetRemoved,'o-', label='Line Profile')
    plt.legend(loc='best')
    plt.xlabel('distance [mm]')
    plt.ylabel('Intensity [cm^1]')
    plt.title('Line Profile for FWHM')
    plt.savefig(OUTPUT_FOLDER + 'FWHM_LineProfile')
    
    FWHM = x1-x0 - ZST_BB_DIAMETER_MM
    print('\nThe FWHM [mm]: {:.3f}'.format(FWHM))
    
    #plt.show()

if __name__ == '__main__':
   main()

