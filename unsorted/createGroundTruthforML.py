import sys
import subprocess
import json
import traceback
import dicom
import nrrd
import numpy as np
import os
from scipy import stats
#from scipy.misc import imsave
#import shutil
import sys
import multiprocessing
import datetime
import time
import logging
import StringIO


def pick(series, axialSlices):
    # seriesOfAcceptableLength = [];
    # for s in range(0,len(series)):
    leadingSeries = -1;
    leadingSeriesZExtent = 1000000;
    # print(axialSlices)

    # pick the first rightly size series
    for s in range(0, len(series)):
        if series[s]:
            # message = "Checking series {}/{}".format(s,len(series))
            # print(message + "\r")
            # sort the series according to the instance numbers
            extractedSeries = axialSlices[(axialSlices[:, 3] == series[s]), :]
            instanceNumber = extractedSeries[:, 4]
            # print("Extracted series length {}".format(len(extractedSeries)))
            if (len(extractedSeries) >= 2):
                if not (np.any(instanceNumber == "")):
                    indices = extractedSeries[:, 4].astype(np.int).argsort()
                    extractedSeries = extractedSeries[indices, :]
                    # print("zPos1 {}, zPos2 {}".format(extractedSeries[0,11],extractedSeries[1,11]))
                    currentSeriesZExtent = len(extractedSeries) * abs(
                        float(extractedSeries[0, 11]) - float(extractedSeries[1, 11]))
                    distanceFrom450Extent = abs(450 - currentSeriesZExtent)
                    # print("Examining series {}, zExtent: {}, distanceFrom450Extent:{}".format(series[s], currentSeriesZExtent, distanceFrom450Extent));
                    if (leadingSeriesZExtent > distanceFrom450Extent):
                        if (currentSeriesZExtent > 225) and (currentSeriesZExtent < 900):
                            # print("Leading series: {}".format(series[s]))
                            leadingSeriesZExtent = distanceFrom450Extent
                            leadingSeries = series[s]
                            # else:
                            # print("Warning: would have taken series {}, however extent is {}".format(series[s], currentSeriesZExtent))
            else:
                print("Warning: series does not have instance numbers, Series:", series[s])
    if (leadingSeries == -1):
        return []
    extractedSeries = axialSlices[(axialSlices[:, 3] == leadingSeries), :]
    
    indices = extractedSeries[:, 4].astype(np.int).argsort()
    extractedSeries = extractedSeries[indices, :]
    currentIndex = extractedSeries[0, 4].astype(np.int)
    for i in range(1, len(extractedSeries)):
        currentIndex = currentIndex + 1
        if (extractedSeries[i, 4].astype(np.int) != currentIndex):
            extractedSeries = []
            print("Warning: missing image {}, from series {}, rejecting series!".format(currentIndex, series[s]))
            break
        if (extractedSeries[i, 12] != extractedSeries[0, 12] and extractedSeries[i, 13] != extractedSeries[0, 13]):
            extractedSeries = []
            print("Warning: mismatch dims, rejecting series!")
            break

    return extractedSeries

def move(pixel_array, spacing, res, offset_mm):
    
    # Note: We are purposely warping in the z direction, no rolling in that dimension will occur
    
    fullVolume_spacing = [450/spacing[0]*res[0], 450/spacing[1]*res[1], float(450)/res[2]]
    fullVolume = np.zeros((int(np.ceil(450/spacing[0])), int(np.ceil(450/spacing[1])), res[2]),dtype='int16') 
    fullVolume_spacing = [float(450)/fullVolume.shape[0], float(450)/fullVolume.shape[1], float(450)/res[2]]
    fullVolume[:res[0], :res[1], :] = pixel_array[:,:,:]
    
    # Move spine to the bottom of the volume
    axisOneShiftmm = 450 - spacing[0]*res[0]
    axisOneShiftRows = axisOneShiftmm/fullVolume_spacing[0]
    # we can't roll partial rows, ensure it's a whole number
    axisOneShiftRows = np.round(axisOneShiftRows)
    fullVolume = np.roll(fullVolume, int(axisOneShiftRows), axis=1)
    
    # Move the spine to the center of the volume
    axisZeroShiftmm = 450 - spacing[1]*res[1]
    axisZeroShiftCols = axisZeroShiftmm/fullVolume_spacing[1]
    axisZeroShiftCols = axisZeroShiftCols / float(2)
    # we can't roll partial rows, ensure it's a whole number
    axisZeroShiftCols = np.round(axisZeroShiftCols)
    fullVolume = np.roll(fullVolume,int(axisZeroShiftCols),axis=0)
    
    fullVolume[:, :int(axisOneShiftRows), :] = 0
    
    # Move the spine closer to the panel
    #offset_rows = offset_mm/fullVolume_spacing[0]
    # we can't roll partial rows, ensure it's a whole number
    #offset_rows = np.round(offset_rows);
    #translated_array = np.roll(fullVolume,int(offset_rows),axis=1)
    #translated_array[:, :int(offset_rows), :] = 0
    
    return fullVolume

def createGroundTruth(study):

    try:
        HOME = os.path.expanduser('~')
        tempFolder = (HOME+'/ML/tempFolder/')
        studyname = study.replace("/\n","")
        start = datetime.datetime.now()
        print("Started processing for file : {}".format(studyname))

        if not os.path.exists(tempFolder):
           os.mkdir(tempFolder)
        ########### 1. name of the study
        #TODO:
        # use only when reading from a file
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
        headerFile = tempFolder+studyname+"_header.txt"
        subprocess.check_output(['aws', 's3api', 'get-object', '--bucket', 'nviewdatasets', '--key',  keyFirst, '--range',
                     'bytes=0-2000' '--output', headerFile])
        ds = dicom.read_file(headerFile)
        os.remove(headerFile)

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
            x = np.zeros([len(keys), number_of_tags], dtype='U200')
            axialSlices = np.zeros([0, number_of_tags], dtype='U200')
            x[0, :] = ["StudyID", "SOPInstanceUID", "StudyDescription", "SeriesNumber", "InstanceNumber",
                       "PatientOrientation", "ImageOrientationPatient", "RescaleSlope", "RescaleIntercept", "KVP",
                       "ImageType", "ImagePositionPatient", "Rows", "Columns"]

            # download headers and accumulate the dicom info
            # print ("Accumulating dicom information")
            SlopeWarning = False
            InterceptWarning = False
            zPosWarning = False
            imageCount = len(keys)
            if (imageCount > 400):
                print("Warning: image count > 400 ({}), only checking first 400".format(imageCount))
                imageCount = 400
            for k in range(0, imageCount):
                # for k in range(0,10):
                # TODO download everything in parallel, syncronize, and then read over each file
                subprocess.check_output(
                    ['aws', 's3api', 'get-object', '--bucket', 'nviewdatasets', '--key', keys[k]['Key'], '--range',
                     'bytes=0-2000' '--output ', headerFile])
                ds = dicom.read_file(headerFile, force=True)
                # print("image dims: {} x {}".format(int(ds.Rows), int(ds.Columns)))
                x[k, 0] = study[:-1]
                if "SOPInstanceUID" in ds:
                    x[k, 1] = ds.SOPInstanceUID
                try:
                    if "StudyDescription" in ds:
                        x[k, 2] = ds.StudyDescription
                except UnicodeDecodeError:
                    x[k, 2] = "ERROR"
                if "SeriesNumber" in ds:
                    x[k, 3] = ds.SeriesNumber
                if "InstanceNumber" in ds:
                    x[k, 4] = ds.InstanceNumber
                if "PatientOrientation" in ds:
                    x[k, 5] = (" ".join(ds.PatientOrientation))
                else:
                    x[k, 5] = ("NONE")
                if "ImageOrientationPatient" in ds:
                    x[k, 6] = (" ".join(map(str, ds.ImageOrientationPatient)))
                if "RescaleSlope" in ds:
                    x[k, 7] = (ds.RescaleSlope)
                else:
                    x[k, 7] = 1
                    SlopeWarning = True
                if "RescaleIntercept" in ds:
                    x[k, 8] = (ds.RescaleIntercept)
                    # print("{}, intercept: {}, slope: {}".format(k, x[k,8], x[k,7]))
                else:
                    x[k, 8] = 0
                    InterceptWarning = True
                if "KVP" in ds:
                    x[k, 9] = (ds.KVP)
                if "ImageType" in ds:
                    x[k, 10] = (" ".join(ds.ImageType))
                if ("ImagePositionPatient") in ds:
                    x[k, 11] = ds.ImagePositionPatient[2]
                    # print(x[k,11]);
                else:
                    x[k, 11] = 0
                    zPosWarning = True
                if ((x[k, 5] == "L P") or ("AXIAL" in x[k, 10])):
                    axialSlices = np.vstack([axialSlices, x[k, :]])
                if ("Rows") in ds:
                    x[k, 12] = ds.Rows
                if ("Columns") in ds:
                    x[k, 13] = ds.Columns
            if(SlopeWarning):
                print ("File: {} : Warning: RescaleSlope not present, Using default value: 1".format(studyname))

            if(InterceptWarning):
                print ("File: {} : Warning: RescaleIntercept not present, Using default value: 0".format(studyname))
            if(zPosWarning):
                print ("File: {} : Warning: ImagePositionPatient not present, Using default value: 0".format(studyname))

            ########### 5. identify all the axial slices
            extractedSeries = []
            isValidAxialSeries = False
            if (len(axialSlices)>0):
                # Series is the list of series in the study, usually 1-5 -JW
                # This is looking for unique series numbers -JW
                series = (np.unique(axialSlices[1:,3]))
                extractedSeries = pick(series, axialSlices)
                if (len(extractedSeries) > 0):
                    isValidAxialSeries = True

            if not isValidAxialSeries:
                print ("File: {} : Warning: No valid axial series found".format(studyname))

            ######### 6. Download all the axialslices and create nrrd for the series
            if(isValidAxialSeries):
                print "Using series {}, # of images = {}".format(extractedSeries[0], len(extractedSeries))
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

                #NOTE: WE ARE FORCING THE SPACING TO WORK OUT TO EXTENT 450 -JW
                #spacing = [float(450)/float(tempSlice1.Rows), float(450)/float(tempSlice1.Columns)]
                spacing = [tempSlice1.PixelSpacing[1], tempSlice1.PixelSpacing[0]]
                
                # Handle volumes with an extent of more than 450
                if (spacing[0] * float(tempSlice1.Columns) > 450):
                    spacing[0] = float(450)/float(tempSlice1.Columns)
                    print("File: {} : Warning: Extent x ({}) larger than 450 detected, forcing to 450".format(studyname, spacing[0] * float(tempSlice1.Rows)))
                if (spacing[1] * float(tempSlice1.Rows) > 450):
                    spacing[1] = float(450)/float(tempSlice1.Rows)
                    print("File: {} : Warning: Extent y ({}) larger than 450 detected, forcing to 450".format(studyname, spacing[1] * float(tempSlice1.Columns)))
                    
                    # Check if the image extent is greater than 150, throw it out if it's not
                if (spacing[1] * float(tempSlice1.Rows) > 150 and spacing[0] * float(tempSlice1.Columns) > 150):

                    #NOTE: WE ARE FORCING THE SPACING TO WORK OUT TO EXTENT 450 -JW
                    zspacing = float(450.0)/float(len(extractedSeries))

                    # create a empty array to store all the converted slices
                    series_image = np.zeros((tempSlice1.pixel_array.shape[1], tempSlice1.pixel_array.shape[0], len(extractedSeries)),dtype='int16')

                    for s in range(0,len(extractedSeries)):
                        objectID = extractedSeries[s,1]
                        objectKey = studyBucket+objectID

                        # download slice and extract pixel_array from dicom data
                        subprocess.check_output(['aws', 's3', 'cp', objectKey, tempFolder])
                        ds = dicom.read_file(tempFolder+objectID)
                        print (objectID)
                        mode = stats.mode(ds.pixel_array, axis=None)

                        I = np.transpose(ds.pixel_array)
                        I = I.view('int16')

                        # eliminate outliers
                        #I = np.clip(I, mode[0], I.max())
                        #mask = I != mode[0]

                        #CT to Hounsfields Units
                        I=a[s]*I+b[s]
                        #I = I*mask

                        # linear attenuation
                        I = (I * 0.0001848 + 0.185)


                        I = I*8192#*mask
                        seri es_image[:,:,s]= I
                        os.remove(tempFolder+objectID)

                    resolution = [series_image.shape[0], series_image.shape[1], series_image.shape[2]]
                    series_image = move(series_image, spacing, resolution, float(20))

                    # clipping negative values
                    mu = series_image
                    if (mu.max() < 10000):
                        print "File: {} : {:%Y-%m-%d %H:%M:%S}, used(BP: {}),series {}, min/max {}/{}".format(studyname, start, bodyPart, extractedSeries[0,3], series_image[0,0,:].min(), series_image[0,0,:].max())

                        # Don't ask my why, but this works !!!
                        mu = np.clip(mu, 0.0, float(2**16-1)) #Clip max to be between 0 and max of ushort
                        # Buuhhh to python

                        mu = mu.astype('uint16')

                        # save the series in s3
                        seriesName = study[:-1]+".nrrd"
                        options={}
                        options["spacings"]=[float(spacing[0]),float(spacing[1]),zspacing]
                        nrrd.write(seriesName, mu, options)

                        # TODO: replace resampling using functions from scikit-learn.
                        subprocess.check_output(['teem-unu', 'resample', '-s',  str(256), str(256), str(256), '-i', seriesName, '-o', seriesName])
                        subprocess.check_output(['teem-unu', 'slice', '-i', seriesName, '-o', str(os.getpid())+".nrrd", '-a', str(2), '-p', str(127)]) 
                        subprocess.check_output(['teem-unu', 'quantize', '-b', str(16), '-i', str(os.getpid())+".nrrd", '-o', study[:-1]+".png"])
                        #TODO: This has to match the volume size in volume geometry section in reconstruction.conf
                        subprocess.check_output(['aws', 's3', 'cp', seriesName, 's3://nviewdatasets/CT_datasets/'])
                        os.remove(seriesName)
                    else:
                        print "Warning: images too bright! removing {}".format(studyname)
                        print "File: {} : {:%Y-%m-%d %H:%M:%S}, unused(bp: {})".format(studyname, start, bodyPart)
                        with open("reject"+str(os.getpid())+".txt",'a') as rejectFile:
                            rejectFile.write(studyname+"\n")
                            rejectFile.close()
                else:
                    print "Warning: extent too small! removing {}".format(studyname)
                    print "File: {} : {:%Y-%m-%d %H:%M:%S}, unused(bp: {})".format(studyname, start, bodyPart)
                    with open("reject"+str(os.getpid())+".txt",'a') as rejectFile:
                        rejectFile.write(studyname+"\n")
                        rejectFile.close()
            else:
                print "File: {} : {:%Y-%m-%d %H:%M:%S}, unused(bp: {})".format(studyname, start, bodyPart)
                with open("reject"+str(os.getpid())+".txt",'a') as rejectFile:
                    rejectFile.write(studyname+"\n")
                    rejectFile.close()

        else:
           print "File: {} : {:%Y-%m-%d %H:%M:%S}, unused(bp: {})".format(studyname, start, bodyPart)
           with open("reject"+str(os.getpid())+".txt",'a') as rejectFile:
                rejectFile.write(studyname+"\n")
                rejectFile.close()
    except Exception as e:

        # Catch the exception and log the stack trace
        logging.error("----------------------------------------------------------------------------------------------------------------")
        exc_buffer = StringIO.StringIO()
        traceback.print_exc(file=exc_buffer)
        logging.error('File: ' + study + ': Uncaught exception in worker process:\n%s',exc_buffer.getvalue())
        logging.error("----------------------------------------------------------------------------------------------------------------")
        pass


def test(str1):

    try:
        print("Starting Thread:")
        print(1/str1)
        print("Exiting Thread: ")
    except Exception as e:
        print("Thread:"+ str(str1))
        #logging.info("Exception :"+str)
        logging.error("----------------------------------------------------------------------------------------------------------------")
        exc_buffer = StringIO.StringIO()
        traceback.print_exc(file=exc_buffer)
        logging.error(str(str1) + ': Uncaught exception in worker process:\n%s',exc_buffer.getvalue())
        logging.error("----------------------------------------------------------------------------------------------------------------")
        pass


def main():

    procs = 64
    # temp output folder to store all the downloaded files
    #tempFolder = str(sys.argv[1])

    # index to start the process
    startIndex = int(sys.argv[1])  # CA: this needs to be a string of the last study, not the index.

    # list out all the studies in nviewdatasets/CT_Scan
    # process = subprocess.Popen(['aws', 's3', 'ls', 's3://nviewdatasets/CT_scans/'],stdout = subprocess.PIPE)
    # studyList = process.communicate()[0]
    # studyList = studyList.replace('PRE','')s
    # studyList = studyList.split()
    # TODO: temporarily loading all the files from a text file
    HOME = os.path.expanduser('~')
    studyList = open(HOME+'/rt3d/ML/listOfStudiesinS3.txt').readlines()
    studyList = studyList[startIndex:]
    #createGroundTruth("1.2.826.0.1.3680043.2.403.1.10.36.177.33.2887505.20150810084500/\n")
    #def adder(value):
    #    global usedStudies
    #    usedStudies += value
    p = multiprocessing.Pool(procs)
    #Hello = [1,0,2,0,5]
    #p.map(test,Hello)
    p.map(createGroundTruth, studyList)
    p.close()
    p.join()

    print("Complete.")

if __name__ == "__main__" :
    main()

