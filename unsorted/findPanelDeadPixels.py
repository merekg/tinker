# -*- coding: utf-8 -*-
import nrrd
import sys
import numpy as np
import matplotlib.pyplot as plt

deadPixelList = [] #List to save the dead pixels 

def findFlickeringPixles(image) :
    
    global deadPixelList

    
    image = np.swapaxes(image, 0, 2)
    
    fliceringPixel = []

    
    print(image.shape)
    
    
    pixelValueAcrossSlices = []
    for i in range(0,len(image[0])) :
        for j in range(0,len(image[0][0])) :
            for z in range(0,len(image)) :
                pixelValueAcrossSlices.append(image[z, j,i])            
            if(max(pixelValueAcrossSlices) - min(pixelValueAcrossSlices) > 2000) :
               fliceringPixel.append([max(pixelValueAcrossSlices), min(pixelValueAcrossSlices), max(pixelValueAcrossSlices) - min(pixelValueAcrossSlices), i, j])
            pixelValueAcrossSlices = []
            
            
    # Write the dead pixels onto a test file
    testfile = open('testFile.txt','w')
    for item in fliceringPixel :
        testfile.write("%s\n" % item)
    testfile.close()        
    plt.imshow(image[1,:,:], cmap='gray')
    plt.show()
    
    
    
''' 
Find the black pixels in the volume
'''    
def findNonResponsivePixels(image) :
   
    global deadPixelList
    
    for i in range(0,len(image)) :
        for j in range(0,len(image[i])) :
            pixelValue = np.asarray(image[i][j],dtype=np.float32)[0]
            if pixelValue == 0.0 or pixelValue >= 65000 :
                if [i,j,pixelValue] not in deadPixelList :
                    deadPixelList.append([i,j])

''' 
Compute the ratio between the means of the two volumes and 
the pixel intensities of the two volumes. The ratio ideally 
needs to be 1. If the pixel's intensities deviate from 1 by 
more than an acceptable threshold we add that pixel to the
dead Pixel List
'''                   
def findDeadPixels(lowIntImage, highIntImage) :

    global deadPixelList
    
    # Compute the mean ratio between the two volumes
    meanLowIntImage =  np.mean(lowIntImage)
    meanHighIntImage =  np.mean(highIntImage)
    meanRatio = meanHighIntImage/meanLowIntImage
    
    for i in range(0,len(lowIntImage)) :
        for j in range(0,len(lowIntImage[i])) :
            pixelValueLow = np.asarray(lowIntImage[i][j])[0]
            pixelValueHigh = np.asarray(highIntImage[i][j])[0]

            if pixelValueLow!= 0:
                # Find the ratio of pixel intensities
                pixelRatio = pixelValueHigh/pixelValueLow
                
                # Find the ratio between mean ratio and pixel intensity ratio.
                finalRatio = meanRatio/pixelRatio
                if finalRatio > 1.22 or finalRatio < 0.8:
                    deadPixelList.append([i,j])
                    
                                  
def main():
    lowIntFileName = str(sys.argv[1])
    highIntFileName = str(sys.argv[2])
    
    # Read the low and high intensity nrrds
    lowIntImage, optionslowIntImage = nrrd.read(lowIntFileName)
    highIntImage, optionshighIntImage = nrrd.read(highIntFileName)

    lowIntImage = lowIntImage.astype(np.float32)
    highIntImage = highIntImage.astype(np.float32)
    
    # Find zeroed pixels 
    findNonResponsivePixels(lowIntImage)
    findNonResponsivePixels(highIntImage)
    
    # Find pixels that are not responsive
    findDeadPixels(lowIntImage, highIntImage)
    
    #Find pixels that flicker TODO: may have been obselete. Need to confirm and delete
    #findFlickeringPixles(lowIntImage.astype(np.float32))
    
    # Write the dead pixels onto a test file
    testfile = open('deadPixMap.txt','w')
    for item in deadPixelList :
        testfile.write("%s\n" % item)
    testfile.close()
    
    
if __name__ == "__main__":
    main()