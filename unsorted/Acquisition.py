import sys
import h5py
import numpy as np

class Acquisition:

    # HDF5 File Paths
    ITKIMAGE_PATH = 'ITKImage/0/'
    METADATA_PATH = ITKIMAGE_PATH + 'MetaData/'
    DIMENSION_PATH = ITKIMAGE_PATH + 'Dimension'
    DIRECTIONS_PATH = ITKIMAGE_PATH + 'Directions'
    ORIGIN_PATH = ITKIMAGE_PATH + 'Origin'
    SPACING_PATH = ITKIMAGE_PATH + 'Spacing'
    VOXELDATA_PATH = ITKIMAGE_PATH + 'VoxelData'
    VOXELTYPE_PATH = ITKIMAGE_PATH + 'VoxelType'
    PMCAL_PATH = METADATA_PATH + 'PMCal'
    METADATATABLE_PATH = METADATA_PATH + 'metaDataTable'
    PMCAL_PATH = METADATA_PATH + 'PMCal'
    PNO_PATH = METADATA_PATH + 'imagePnO'

    # Attribute names
    SCALE_ATTRIBUTE = 'scale'
    PMCAL_ATTRIBUTE = 'nAngles'
    MOTOR_SPEED = 'MotorSpeed'
    NUMBER_IMAGES = 'nImages'
    TYPE_ATTRIBUTE = 'Type'

    def __init__(self, filePath):

        # Open acquisition file and load data
        with h5py.File(filePath, 'r') as acquisitionFile:

            # Get scale from the voxel data
            self._scale = float(acquisitionFile[self.VOXELDATA_PATH].attrs[self.SCALE_ATTRIBUTE])

            # Get data from file
            self._images = acquisitionFile[self.VOXELDATA_PATH][:] / self._scale
            self._spacing = acquisitionFile[self.SPACING_PATH][:]
            self._metaDataTable = acquisitionFile[self.METADATATABLE_PATH][:]
            self._projectionMatricies = acquisitionFile[self.PMCAL_PATH][:]
            self._poseAndOrientation = acquisitionFile[self.PNO_PATH][:]

            # Get additional attributes
            self.nAngles = acquisitionFile[self.PMCAL_PATH].attrs[self.PMCAL_ATTRIBUTE]
            self._motorSpeed = acquisitionFile[self.METADATATABLE_PATH].attrs[self.MOTOR_SPEED]
            self.imageType = acquisitionFile[self.VOXELDATA_PATH].attrs[self.TYPE_ATTRIBUTE]
            #for attribute in acquisitionFile[self.VOXELDATA_PATH].attrs:
                #self._imageAttributes[attribute] = acquisitionFile[self.VOXELDATA_PATH].attrs[attribute]

    def saveAsFloat(self, filePath):

        # Open output file and save out data
        with h5py.File(filePath, 'w') as outputFile:

            dimensions = list(self._images.shape)
            imageDimension = dimensions[1:3] + dimensions[0:1] 
            #imageDimension = dimensions # HACK: doesn't match our volume format

            # Write out acquisition data
            outputFile.create_dataset(self.VOXELDATA_PATH, data=self._images, dtype=np.single)
            outputFile.create_dataset(self.VOXELTYPE_PATH, data="float")

            # Fill in ITK requred data fields
            outputFile.create_dataset(self.DIMENSION_PATH, data=imageDimension)
            outputFile.create_dataset(self.DIRECTIONS_PATH, data=np.identity(3))
            outputFile.create_dataset(self.ORIGIN_PATH, data=np.zeros(3))
            outputFile.create_dataset(self.SPACING_PATH, data=self._spacing)

            # Fill in MetaData fields
            outputFile.create_dataset(self.PMCAL_PATH, data=self._projectionMatricies)
            outputFile.create_dataset(self.PNO_PATH, data=self._poseAndOrientation)
            outputFile.create_dataset(self.METADATATABLE_PATH, data=self._metaDataTable)

            # Write additional attributes
            #outputFile[self.VOXELDATA_PATH].attrs.create(self.SCALE_ATTRIBUTE, str(self._scale))
            outputFile[self.PMCAL_PATH].attrs.create(self.PMCAL_ATTRIBUTE, self.nAngles)
            outputFile[self.METADATATABLE_PATH].attrs.create(self.MOTOR_SPEED, self._motorSpeed)
            outputFile[self.VOXELDATA_PATH].attrs.create(self.NUMBER_IMAGES, str(1))
            outputFile[self.VOXELDATA_PATH].attrs.create(self.SCALE_ATTRIBUTE, str(1.000000))
            outputFile[self.VOXELDATA_PATH].attrs.create(self.TYPE_ATTRIBUTE, self.imageType)
            #for attribute in self._imageAttributes:
                #outputFile[self.VOXELDATA_PATH].attrs.create(attribute, self._imageAttributes[attribute])

    def deleteImage(self, index):

        # Get data from file
        self._images = np.delete(self._images, index, 0)
        self._metaDataTable = np.delete(self._metaDataTable, index, 0)

    def deleteImagesBefore(self, index):

        # Drop all images before index
        self._images = self._images[index:]
        self._metaDataTable = self._metaDataTable[index:]

    def deleteImagesAfter(self, index):

        # Drop all images after index
        self._images = self._images[:index]
        self._metaDataTable = self._metaDataTable[:index]

    def append(self, other):

        # Combine images and acquisitions
        self._images = np.concatenate((self._images, other._images), axis=0)
        self._metaDataTable = np.concatenate((self._metaDataTable, other._metaDataTable), axis=0)

    def averageAndNormalizeImages(self, axis):
        avg = np.mean(self._images)
        self._images = np.mean(self._images, axis=axis)/avg
        self._images = np.swapaxes(np.transpose(self._images[:,:,None]),1,2)
        print self._images.shape

def main():

    # Get file paths for acquisition and coefficients from arguments
    inputFilePath = sys.argv[1]
    outputFilePath = sys.argv[2]

    # Open input file, get image stack, and create output file
    acquisition = Acquisition(inputFilePath) 
    acquisition.save(outputFilePath)
    #imageStack = acquisition._images

    # Average images and Angles together
    #imageStack = [np.mean(imageStack[i:i+N], axis=0) for i in range(0, len(imageStack), N)]
    #imageStack = np.array(imageStack)

    # Update metadata for averaged image stack
    #metaDataTable = inputFile[METADATATABLE_PATH][:]
    #metaDataTable = np.array([list(row) for row in metaDataTable])
    
    # Wrap angles in metadataTable, average N items in table, and unwrap anbles
    #metaDataTable[:,0] += 360.0
    #metaDataTable = np.array([np.mean(metaDataTable[i:i+N], axis=0) for i in range(0, len(metaDataTable), N)])
    #metaDataTable[:,0] %= 360.0

    # Create output file and copy over all but voxel data
    #with h5py.File(outputFilePath, 'w') as outputFile:

        # Create group for dataset
        #outputFile.create_group(HDF5_IMAGE_PATH)

        # Iterate over items in image and copy all except for the voxel data
        #for item in inputFile[HDF5_IMAGE_PATH]:
            #if item not in ['Dimension', 'MetaData', 'VoxelData']:
                #inputFile.copy(HDF5_IMAGE_PATH + item, outputFile[HDF5_IMAGE_PATH])

        # Replace image stack and metaDatA
        #outputFile.create_dataset(VOXELDATA_PATH, data=imageStack)
        #outputFile[VOXELDATA_PATH].attrs.create('scale', scaleFactorStr)
        #outputFile.create_dataset(METADATATABLE_PATH, data=metaDataTable)
        #outputFile.create_dataset(DIMENSION_PATH, data=imageStack.shape)

if __name__ == '__main__':
    main()
