import sys
import subprocess
import json
import string
import dicom
import nrrd
import numpy as np
import os
from scipy import stats
from scipy.misc import imsave
import shutil
import sys
import datetime

# temp output folder to store all the downloaded files
tempFolder = str(sys.argv[1])

#input file for the acquisition
singleInitial = str(sys.argv[2])


# index to start the process
startIndex = int(sys.argv[3]) # CA: this needs to be a string of the last study, not the index.


# list out all the studies in nviewdatasets/CT_Scan
#process = subprocess.Popen(['aws', 's3', 'ls', 's3://nviewdatasets/CT_scans/'],stdout = subprocess.PIPE)
#studyList = process.communicate()[0]
#studyList = studyList.replace('PRE','')
#studyList = studyList.split()
#TODO: temporarily loading all the files from a text file 
studyFile = open('/home/justina/rt3d/rt3d/ML/listOfStudiesinS3.txt') 
studyList =  studyFile.readlines()


usedStudies = 0
# change acl of all the studies
##TODO: change to length of full list

def pick(series, axialSlices):
    #seriesOfAcceptableLength = [];
    #for s in range(0,len(series)):
    leadingSeries = -1;
    leadingSeriesZExtent = 1000000;
    #print(axialSlices)

    # pick the first rightly size series
    for s in range(0,len(series)):
        if series[s]:
            #message = "Checking series {}/{}".format(s,len(series))
            #print(message + "\r")
            # sort the series according to the instance numbers
            extractedSeries = axialSlices[ (axialSlices[:,3]==series[s]),:]
            instanceNumber = extractedSeries[:,4]
            #print("Extracted series length {}".format(len(extractedSeries)))
            if (len(extractedSeries) >= 2):
                if not (np.any(instanceNumber == "")): 
                    indices = extractedSeries [:,4].astype(np.int).argsort()
                    extractedSeries  = extractedSeries [indices,:]
                    #print("zPos1 {}, zPos2 {}".format(extractedSeries[0,11],extractedSeries[1,11]))
                    currentSeriesZExtent = len(extractedSeries) * abs(float(extractedSeries[0,11]) - float(extractedSeries[1,11]))
                    distanceFrom450Extent = abs(450-currentSeriesZExtent)
                    #print("Examining series {}, zExtent: {}, distanceFrom450Extent:{}".format(series[s], currentSeriesZExtent, distanceFrom450Extent));
                    if (leadingSeriesZExtent > distanceFrom450Extent):
                        if (currentSeriesZExtent > 225) and (currentSeriesZExtent < 900):
                            #print("Leading series: {}".format(series[s]))
                            leadingSeriesZExtent = distanceFrom450Extent
                            leadingSeries = series[s]
                        #else:
                            #print("Warning: would have taken series {}, however extent is {}".format(series[s], currentSeriesZExtent))
            else:
                print("Warning: series does not have instance numbers, Series:", series[s])
    if (leadingSeries == -1):
        return []
    extractedSeries = axialSlices[ (axialSlices[:,3]==leadingSeries),:]
    indices = extractedSeries [:,4].astype(np.int).argsort()
    extractedSeries  = extractedSeries [indices,:]
    currentIndex = extractedSeries[0,4].astype(np.int)
    for i in range(1, len(extractedSeries)):
		currentIndex = currentIndex + 1
		if (extractedSeries[i,4].astype(np.int) != currentIndex):
			extractedSeries = []
			print("Warning: missing image {}, from series {}, rejecting series!".format(currentIndex,series[s]))
			break
        if (extractedSeries[i,12] != extractedSeries[0,12] and extractedSeries[i,13] != extractedSeries[0,13]):
		    extractedSeries = []
			print("Warning: mismatch dims, rejecting series!")
			break
		
    return extractedSeries

def move(pixel_array, spacing, offset_mm):
	offset_rows = offset_mm/spacing
	# we can't roll partial rows, ensure it's a whole number
	offset_rows = np.round(offset_rows);
	translated_array = np.roll(pixel_array,int(offset_rows),axis=1)
	#translated_array = pixel_array
	translated_array[:, :int(offset_rows), :] = 0
	return translated_array
    
for i in range(startIndex,len(studyList)):

    start = datetime.datetime.now()

    if not os.path.exists(tempFolder):
       os.mkdir(tempFolder)
    ########### 1. name of the study
    #TODO:
    # use only when reading from a file
    study = studyList[i]
    study = study[:-1]
    studyBucket = 's3://nviewdatasets/CT_scans/'+ study

    ########### 2. list all objects in study
    studyPrefix = "CT_scans/" + study
    objectlistCMD = "aws s3api list-objects-v2 --bucket nviewdatasets --prefix " + studyPrefix + " --query '[Contents[].{Key: Key}]'"
    objectsInStudy = subprocess.Popen(objectlistCMD, shell = True, stdout=subprocess.PIPE)
    std_out, std_error = objectsInStudy.communicate()
    data = json.loads(std_out)
    keys = data[0]

    # download 1st object in study and check if it belongs to spine study
    keyFirst = keys[0]['Key']
    headerFile = tempFolder+'header.txt'
    subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'nviewdatasets', '--key',  keyFirst, '--range', 'bytes=0-2000' '--output', headerFile])
    ds = dicom.read_file(headerFile)
    if "StudyDescription" in ds:
        studyName = ds.StudyDescription
    else:
        studyName = "not_given"
    if "BodyPartExamined" in ds:
        bodyPart = ds.BodyPartExamined
    else:
        bodyPart = "not_given"

    if   ("tacar" in studyName.lower()) or ("tor" in studyName.lower()) or ("rax" in studyName.lower()) or ("urotac" in studyName.lower()) or ("spin" in studyName.lower()) or ("abdo" in studyName.lower()) or ("domen" in studyName.lower()) or ("spine" in studyName.lower()) or ("rachi" in studyName.lower()) or ("lomb" in studyName.lower()) or ("tacar" in bodyPart.lower()) or ("tor" in bodyPart.lower()) or ("rax" in bodyPart.lower()) or ("urotac" in bodyPart.lower()) or ("spin" in bodyPart.lower()) or ("abdo" in bodyPart.lower()) or ("domen" in bodyPart.lower()) or ("spine" in bodyPart.lower()) or ("rachi" in bodyPart.lower()) or ("lomb" in bodyPart.lower()):
       number_of_tags = 14
       x=np.zeros([len(keys),number_of_tags],dtype='U200')
       axialSlices=np.zeros([0,number_of_tags],dtype='U200')
       x[0,:]=["StudyID","SOPInstanceUID", "StudyDescription", "SeriesNumber", "InstanceNumber", "PatientOrientation", "ImageOrientationPatient", "RescaleSlope", "RescaleIntercept", "KVP", "ImageType", "ImagePositionPatient", "Rows", "Columns"]

       # download headers and accumulate the dicom info
       #print ("Accumulating dicom information")
       SlopeWarning = False
       InterceptWarning = False
       zPosWarning = False
       imageCount = len(keys)
       if (imageCount > 400):
           print("Warning: image count > 400 ({}), only checking first 400".format(imageCount))
           imageCount = 400
       for k in range(0,imageCount):
       #for k in range(0,10):
           # TODO download everything in parallel, syncronize, and then read over each file
           subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'nviewdatasets', '--key',  keys[k]['Key'], '--range', 'bytes=0-2000' '--output ', headerFile])
           ds = dicom.read_file(headerFile, force=True)
           #print("image dims: {} x {}".format(int(ds.Rows), int(ds.Columns)))
           x[k,0] = study[:-1]
           if "SOPInstanceUID" in ds:
              x[k,1]=ds.SOPInstanceUID
           try:
              if "StudyDescription" in ds:
                  x[k,2]=ds.StudyDescription
           except UnicodeDecodeError:
              x[k,2]="ERROR"
           if "SeriesNumber" in ds:
              x[k,3] = ds.SeriesNumber
           if "InstanceNumber" in ds:
              x[k,4]=ds.InstanceNumber
           if "PatientOrientation" in ds:
              x[k,5]=(" ".join(ds.PatientOrientation))
           else:
              x[k,5]=("NONE")
           if "ImageOrientationPatient" in ds:
              x[k,6]=(" ".join(map(str, ds.ImageOrientationPatient)))
           if "RescaleSlope" in ds:
              x[k,7]=(ds.RescaleSlope)
           else:
              x[k,7]=1
              SlopeWarning = True
           if "RescaleIntercept" in ds:
              x[k,8]=(ds.RescaleIntercept)
          #print("{}, intercept: {}, slope: {}".format(k, x[k,8], x[k,7]))
           else:
              x[k,8]=0
              InterceptWarning = True
           if "KVP" in ds:
              x[k,9]=(ds.KVP)
           if "ImageType" in ds:
              x[k,10]=(" ".join(ds.ImageType))
           if ("ImagePositionPatient") in ds:
              x[k,11]=ds.ImagePositionPatient[2]
          #print(x[k,11]);
           else:
              x[k,11]=0
              zPosWarning = True
           if ((x[k,5]=="L P") or ("AXIAL" in x[k,10])):
              axialSlices = np.vstack([axialSlices, x[k,:]])
           if ("Rows") in ds:
              x[k,12] = ds.Rows
           if ("Columns") in ds:
              x[k,13] = ds.Columns
       
       if(SlopeWarning):
           print ("Warning: RescaleSlope not present, Using default value: 1")

       if(InterceptWarning):
           print ("Warning: RescaleIntercept not present, Using default value: 0")
       if(zPosWarning):
           print ("Warning: ImagePositionPatient not present, Using default value: 0")


       ########### 5. identify all the axial slices
       #if ((x[:,5]=="L P") || ("AXIAL" in x[:,10])
       extractedSeries = []
       isValidAxialSeries = False
       if (len(axialSlices)>0): 
          # Series is the list of series in the study, usually 1-5 -JW
          # This is looking for unique series numbers -JW
          series = (np.unique(axialSlices[1:,3]))
          #print("Axial slice count: ", len(axialSlices), ", series count: ", len(series));
          #print(series)
          extractedSeries = pick(series, axialSlices)
          if (len(extractedSeries) > 0):
              isValidAxialSeries = True;
      
       if not isValidAxialSeries:
          print ("Warning: No valid axial series found")

       ######### 6. Download all the axialslices and create nrrd for the series
       if(isValidAxialSeries):
          #print "Using series {}, # of images = {}".format(extractedSeries[0], len(extractedSeries))
          usedStudies += 1
          # extract the slope and intercept for the series, a-slope, b-intercept
          a = (extractedSeries[:,7])
          a = a.astype('int') 
          b = (extractedSeries[:,8])
          b = b.astype('int')
          # Slope and intercept are on a per image basis, not the same for all images.
          # We need to be able to associate to each image the slope and intercept at the beginning and be able to     retrieve it here
          # And Actually. this section of picking a and b should be closer to the point of use, down below.
          
          # extract the mode (using the 1st and 2nd image) and zspacing 
          objectKey = studyBucket+extractedSeries[0,1]
          # TODO download in parallel
          subprocess.check_output(['aws', 's3', 'cp', objectKey, tempFolder])
          tempSlice1 = dicom.read_file(tempFolder + extractedSeries[0,1])

          #mode = stats.mode(tempSlice1.pixel_array,axis = None)

          #if "PixelSpacing" in tempSlice1:
             #spacing = tempSlice1.PixelSpacing
          #else:
             #spacing = [(0.625*512)/tempSlice1.pixel_array.shape[0], (0.625*512)/tempSlice1.pixel_array.shape[1]]
          #NOTE: WE ARE FORCING THE SPACING TO WORK OUT TO EXTENT 450 -JW
          spacing = [float(450)/float(tempSlice1.Rows), float(450)/float(tempSlice1.Columns)]
             #print ("Warning: Resolution not present, Using estimated spacing {}, {}".format(spacing[0],spacing[1]))
          objectKey = studyBucket+extractedSeries[1,1]
          subprocess.check_output(['aws', 's3', 'cp', objectKey, tempFolder])
          tempSlice2 = dicom.read_file(tempFolder + extractedSeries[1,1])
          #if "ImagePositionPatient" in tempSlice1:
              #zspacing = abs(float(tempSlice1.ImagePositionPatient[2]) - float(tempSlice2.ImagePositionPatient[2]))
          #else:
          #NOTE: WE ARE FORCING THE SPACING TO WORK OUT TO EXTENT 450 -JW
          zspacing = float(450.0)/float(len(extractedSeries))
              #print ("Warning: Resolution not present, Using estimated zresolution {}".format(zspacing))
          # create a empty array to store all the converted slices
          series_image = np.zeros((tempSlice1.pixel_array.shape[1], tempSlice1.pixel_array.shape[0], len(extractedSeries)),dtype='int16')

          for s in range(0,len(extractedSeries)):
              objectID = extractedSeries[s,1]
              objectKey = studyBucket+objectID
              # download slice and extract pixel_array from dicom data
              subprocess.check_output(['aws', 's3', 'cp', objectKey, tempFolder])
              ds = dicom.read_file(tempFolder+objectID)
              mode = stats.mode(ds.pixel_array, axis=None)
              I = np.transpose(ds.pixel_array) 
              # eliminate outliers
              I = np.clip(I, mode[0], I.max())
              mask = I != mode[0]
              #CT to Hounsfields Units
              #print("Applied slope ", a[s], " and intercept ", b[s], " to image ", s)
              I=a[s]*I+b[s]
              I = I*mask 
              # linear attenuation
              I = (I * 0.0001848 + 0.185)
              #print('Img {}: Min/max of mx in cm-1 {}/{}'.format(s, I.min(), I.max()))

              I = I*8192*mask
              #print('Img {}: Min/max of mx in cm-1*8192 {}/{}'.format(s, I.min(), I.max()))
              series_image[:,:,s]= I

          print "{:%Y-%m-%d %H:%M:%S}, Used: {}, BP: {}, Current: {}, series {}, min/max {}/{}, {}".format(start, usedStudies, bodyPart, i, extractedSeries[0,3], series_image[0,0,:].min(), series_image[0,0,:].max(), study)

          series_image = move(series_image, float(spacing[0]), float(20))

          # clipping negative values
          mu = series_image
          #print('Min/max of mu pre stauration {}/{}'.format(mu.min(), mu.max()))

          # Don't ask my why, but this works !!!
          mu = np.clip(mu, 0.0, float(2**16-1)) #Clip max to be between 0 and max of ushort
          # Buuhhh to python

          #print('Min/max of mu post stauration {}/{}'.format(mu.min(), mu.max()))
          mu = mu.astype('uint16') 
          #print('Min/max of mu post ushort conversion {}/{}'.format(mu.min(), mu.max()))

          #print('Min/max of mu {}/{}'.format(mu.min(), mu.max()))
          # save the series in s3
          seriesName = study[:-1]+".nrrd"  
          #rawName = study[:-1]+".raw.gz" 
          options={}
          #print("spacings, {} x {} x {}".format(float(spacing[0]),float(spacing[1]),zspacing))
          options["spacings"]=[float(spacing[1]),float(spacing[0]),zspacing]
          nrrd.write(seriesName, mu, options)
          nrrd.write("debug."+seriesName, mu, options)

          # resampling to make the number of slices a multiple of 8
          #if (mu.shape[2]%8!=0):
          #      resampleZ = (mu.shape[2]/8 + 1) * 8
          # TODO: replace resampling using functions from scikit-learn.
          subprocess.check_output(['teem-unu', 'resample', '-s',  str(256), str(256), str(256), '-i', seriesName, '-o', seriesName])  
          #TODO: This has to match the volume size in volume geometry section in reconstruction.conf
          #subprocess.check_output(['teem-unu', 'save', '-f', 'nrrd', '-e', 'gzip', '-i', seriesName, '-o', seriesName]) 
 


          #subprocess.check_output(['aws', 's3', 'cp', rawName,'s3://nviewdatasets/CT_datasets/'])
          
          #Launch the Reconstruction and Acquisition
          acqMode = open(singleInitial)
          acqProcess = subprocess.Popen(['AcquisitionSimulation','-v', '0', '-c', '~/AcqSim.conf', '-i', seriesName , '-q'],stdin=acqMode)
          
          # TODO: Need to update the config file for every study
          reconProcess = subprocess.Popen(['Reconstruction','-v', '0', '-c', '~/ReconSim.conf', '-l', '-q'])
          reconOutput = reconProcess.communicate()
          # delete all the downloaded slices
          #shutil.rmtree(tempFolder) #TODO Nisha, this command could erase a folder, is this what created the lost folder issue? 
                    # TODO: for safety, we should not rmtree

          reconName = "recon"+study[:-1]+".nrrd" 
          # rename the file at the default saved location
 
          subprocess.check_output(['mv', '/media/ramdisk/recon0.nrrd', reconName])

          # quantize the input volume (8-bit)
          #subprocess.check_output("teem-unu quantize -b 8 -min 0 -max 255 -i {} | teem-unu convert -t uchar -o {}".format(seriesName, seriesName), shell=True)

      #subprocess.check_output(['teem-unu', 'resample', '-s',  str(256), str(256), str(256), '-i', reconName, '-o', reconName])  

          # quantize the output volume (8-bit)
          #subprocess.check_output("teem-unu quantize -b 8 -min 0 -max 255 -i {} | teem-unu convert -t uchar -o {}".format(reconName,reconName), shell=True)
          subprocess.check_output("teem-unu quantize -b 16 -min 0 -max {} -i {} | teem-unu convert -t uint16 -o {}".format(float(2**16-1),reconName,reconName), shell=True)
          
          # upload the series converted .nrrd format on the cloud                        
          subprocess.check_output(['aws', 's3', 'cp', seriesName,'s3://nviewdatasets/CT_datasets/'])
          subprocess.check_output(['aws', 's3', 'cp', reconName,'s3://nviewdatasets/Recons/'])


         # Remove the input and the reconstructed volumes
          os.remove(seriesName)
          os.remove(reconName)

    else:
       print "{:%Y-%m-%d %H:%M:%S}, Used: {}, Current: {}, unused(name: {}, bp: {}), {}".format(start, usedStudies, i, studyName, bodyPart, study)


        

