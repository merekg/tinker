import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

class Acquisition:

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
    SCALE_ATTRIBUTE = 'scale'
    PMCAL_ATTRIBUTE = 'nAngles'
    MOTOR_SPEED = 'MotorSpeed'

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
            self._imageAttributes = {}
            for attribute in acquisitionFile[self.VOXELDATA_PATH].attrs:
                self._imageAttributes[attribute] = acquisitionFile[self.VOXELDATA_PATH].attrs[attribute]

    def save(self, filePath):

        # Open output file and save out data
        with h5py.File(filePath, 'w') as outputFile:

            dimensions = list(self._images.shape)
            imageDimension = dimensions[1:3] + dimensions[0:1] 

            # Write out acquisition data
            self._images = (self._images * self._scale).astype(np.uint16)
            outputFile.create_dataset(self.VOXELDATA_PATH, data=self._images)
            outputFile.create_dataset(self.VOXELTYPE_PATH, data='USHORT')

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
            for attribute in self._imageAttributes:
                outputFile[self.VOXELDATA_PATH].attrs.create(attribute, self._imageAttributes[attribute])

def main():

    # Get file paths for acquisition and coefficients from arguments
    inputFilePath1 = sys.argv[1]
    inputFilePath2 = sys.argv[2]
    #outputFilePath = sys.argv[3]

    acquisition1 = Acquisition(inputFilePath1)
    acquisition2 = Acquisition(inputFilePath2)

    projection1 = acquisition1._images.mean(axis=0) 
    plt.imshow(projection1)
    plt.show()

    projection2 = acquisition2._images.mean(axis=0)
    plt.imshow(projection2)
    plt.show()

    differenceProjection = projection1 - projection2
    plt.imshow(differenceProjection)
    plt.show()
    
if __name__ == '__main__':
    main()
