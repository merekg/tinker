import sys
import h5py
import numpy as np

class Acquisition:

    # HDF5 file paths
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
        self._acquisitionFile = h5py.File(filePath, 'r+')

    def __del__(self):
        self._acquisitionFile.close()

    def getScale(self):
        return float(self._acquisitionFile[self.VOXELDATA_PATH].attrs[self.SCALE_ATTRIBUTE])

    def getScaledImages(self):
        return self._acquisitionFile[self.VOXELDATA_PATH][:] / self.getScale()

    def getRawImages(self):
        return self._acquisitionFile[self.VOXELDATA_PATH][:]

    def getSpacing(self):
        return self._acquisitionFile[self.SPACING_PATH][:]

    def getMetadata(self):
        return self._acquisitionFile[self.METADATATABLE_PATH][:]
    
    def getProjectionMatricies(self):
        return self._acquisitionFile[self.PMCAL_PATH][:]

    def getPose(self):
        return self._acquisitionFile[self.PNO_PATH][:]

    def getDataType(self):
        return self._acquisitionFile[self.VOXELDATA_PATH].attrs[self.TYPE_ATTRIBUTE]

    def getMotorSpeed(self):
        return self._acquisitionFile[self.METADATATABLE_PATH].attrs[self.MOTOR_SPEED]

    def getAllAttributes(self, hdf5Path):
        attributes = {}
        for attribute in self._acquisitionFile[hdf5Path].attrs:
            attributes[attribute] = self._acquisitionFile[hdf5Path].attrs[attribute]
        return attributes

    def writeAllAttributes(self, hdf5Path, attributes):
        for attribute, value in attributes.items():
            self._acquisitionFile[hdf5Path].attrs.create(attribute, value)

    def getPMNumberOfImages(self):
        return self._acquisitionFile[self.PMCAL_PATH].attrs[self.PMCAL_ATTRIBUTE]

    def writeRawImages(self, images):

        # Get existing attributes
        attributes = self.getAllAttributes(self.VOXELDATA_PATH)

        # Delete existng dataset
        del self._acquisitionFile[self.VOXELDATA_PATH]

        # Write out new dataset
        self._acquisitionFile.create_dataset(self.VOXELDATA_PATH, data=images.astype(np.uint16))

        # Write in all previous attributes
        self.writeAllAttributes(self.VOXELDATA_PATH, attributes)

        # Change number of images attribute to new image stack size
        self._acquisitionFile[self.VOXELDATA_PATH].attrs[self.NUMBER_IMAGES] = str(len(images))

        # Update the acquisition file size
        self._acquisitionFile[self.DIMENSION_PATH][2] = len(images)

    def writeMetadata(self, metadata):

        # Get existing attributes
        motorSpeed = self.getMotorSpeed()

        # Delete existing metadata
        del self._acquisitionFile[self.METADATATABLE_PATH]

        # Write out new metadata
        self._acquisitionFile.create_dataset(self.METADATATABLE_PATH, data=metadata)

        # Write out motor speed
        self._acquisitionFile[self.METADATATABLE_PATH].attrs.create(self.MOTOR_SPEED, motorSpeed)

    def copyAs(self, filePath):
        with h5py.File(filePath, 'w') as copyFile:
            for group in self._acquisitionFile:
                self._acquisitionFile.copy(group, copyFile)

    #def saveAs(self, filePath):

        # Open output file and save out data
        #with h5py.File(filePath, 'w') as outputFile:

            #dimensions = list(self._images.shape.astype(np.int32)
            #imageDimension = dimensions[1:3] + dimensions[0:1] 
            #imageDimension = dimensions[::-1] 

            # Write out acquisition data
            #self._images = (self._images * self._scale).astype(np.uint16)
            #outputFile.create_dataset(self.VOXELDATA_PATH, data=self._images)
            #outputFile.create_dataset(self.VOXELTYPE_PATH, data='USHORT')

            # Fill in ITK requred data fields
            #outputFile.create_dataset(self.DIMENSION_PATH, data=imageDimension)
            #outputFile.create_dataset(self.DIRECTIONS_PATH, data=np.identity(3))
            #outputFile.create_dataset(self.ORIGIN_PATH, data=np.zeros(3))
            #outputFile.create_dataset(self.SPACING_PATH, data=self._spacing)

            # Fill in MetaData fields
            #outputFile.create_dataset(self.PMCAL_PATH, data=self._projectionMatricies)
            #outputFile.create_dataset(self.PNO_PATH, data=self._poseAndOrientation)
            #outputFile.create_dataset(self.METADATATABLE_PATH, data=self._metaDataTable)

            # Write additional attributes
            #outputFile[self.VOXELDATA_PATH].attrs.create(self.SCALE_ATTRIBUTE, str(self._scale))
            #outputFile[self.PMCAL_PATH].attrs.create(self.PMCAL_ATTRIBUTE, self._nAngles)
            #outputFile[self.METADATATABLE_PATH].attrs.create(self.MOTOR_SPEED, self._motorSpeed)
            #outputFile[self.VOXELDATA_PATH].attrs.create(self.NUMBER_IMAGES, str(len(self._images)))
            #outputFile[self.VOXELDATA_PATH].attrs.create(self.SCALE_ATTRIBUTE, str(self._scale))
            #outputFile[self.VOXELDATA_PATH].attrs.create(self.TYPE_ATTRIBUTE, self.imageType)
            #for attribute in self._imageAttributes:
                #outputFile[self.VOXELDATA_PATH].attrs.create(attribute, self._imageAttributes[attribute])

def main():

    # Load all input acquisitions
    acquisitions = [Acquisition(path) for path in sys.argv[1:-1]]

    # Copy first input acquisition to output as template
    acquisitions[0].copyAs(sys.argv[-1])

    # Load output acquisition
    combinedAcquisition = Acquisition(sys.argv[-1])

    # Concatenate images for all acquisitions
    combinedImages = np.concatenate([acquisition.getRawImages() for acquisition in acquisitions])

    # Write concatenated images to output file
    combinedAcquisition.writeRawImages(combinedImages)

    # Concatenate metadata for all acquisitions
    combinedMetadata = np.concatenate([acquisition.getMetadata() for acquisition in acquisitions])

    # Write concatenated metadata to output file
    combinedAcquisition.writeMetadata(combinedMetadata)

if __name__ == '__main__':
    main()
