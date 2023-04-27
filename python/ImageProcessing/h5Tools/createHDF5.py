import pymysql
import pandas
import sys, getopt, os
import h5py
import numpy as np

#Global variables

#Database connection variables
HOST = 'localhost'
USER = 'root'
PASSWORD = ''
DATABASE = 'company'
RECONID = ''
fileName = ''
class CreateHDF5:
    def __init__(self, HOST, USER, PASSWORD, DATABASE, RECONID):
        #Establish connection
        self.conn = pymysql.connect(host = HOST, user = USER, password = PASSWORD, database = DATABASE, cursorclass=pymysql.cursors.DictCursor)
        self.reconID = RECONID
    
    def fetchDataFromDatabase(self):
        databaseCursor = self.conn.cursor()
        reconQuery = 'select * from reconstructions where ReconID = ' + self.reconID + ';'
        numberOfRowsAffected = databaseCursor.execute(reconQuery)
        if numberOfRowsAffected == 0:
            print('Failed to fetch reconstruction')
        reconResult = databaseCursor.fetchall()
        reconRow = reconResult[0]        
        self.conn.commit()
        procedureID = str(reconRow['ProcedureID'])
        procedureQuery = 'select * from procedures where ProcedureID = ' + procedureID + ';'
        numberOfRowsAffected = databaseCursor.execute(procedureQuery)
        if numberOfRowsAffected == 0:
            print('Failed to fetch procedure')
        procedureResult = databaseCursor.fetchall()
        procedureRow = procedureResult[0]
        
        return reconRow, procedureRow
    
    def createHDF5(self, fileName, reconRow, procedureRow):
        hdf5File = h5py.File(fileName,'w')
        itkImageGroup = hdf5File.create_group("ITKImage")
        zeroGroup = itkImageGroup.create_group("0")
        
        dimensionsData = np.array([int(reconRow['dimensionX']), int(reconRow['dimensionY']), int(reconRow['dimensionZ'])]) 
        dimensionsDataset = zeroGroup.create_dataset('Dimension', dimensionsData.shape, dtype = int)
        dimensionsDataset[:] = dimensionsData
        
        directionsData = np.zeros((3,3), float)
        np.fill_diagonal(directionsData, 1.0)
        directionsDataset = zeroGroup.create_dataset('Directions', directionsData.shape, dtype = float)
        directionsDataset[:] = directionsData
        
        originData = np.array([0.0,0.0,0.0])
        originDataset = zeroGroup.create_dataset('Origin', originData.shape, dtype = float)
        originDataset[:] = originData
        
        spacingData = np.array([float(reconRow['spacingX']), float(reconRow['spacingY']), float(reconRow['spacingZ'])]) 
        spacingDataset = zeroGroup.create_dataset('Spacing', spacingData.shape, dtype = float)
        spacingDataset[:] = spacingData
        
        print(reconRow['savePath'])
        voxelArray = []
        with open(reconRow['savePath'], 'rb') as f:
            byte = f.read(1)
            while byte != b"":
                # Do stuff with byte.
                voxelArray.append(int.from_bytes(byte, "big"))
                byte = f.read(1)
        voxel = np.asarray(voxelArray, dtype=np.float32)
        voxelData = np.reshape(voxel, (int(reconRow['dimensionX']), int(reconRow['dimensionY']), int(reconRow['dimensionZ'])))
        voxelDataset = zeroGroup.create_dataset('VoxelData', voxelData.shape, dtype = float)
        voxelDataset[:] = voxelData
        voxelDataset.attrs.create('SaveCheck', 1.0)
        voxelDataset.attrs.create('TimeStamp', str(reconRow['acquisitionDate']))
        voxelDataset.attrs.create('nImages', float(reconRow['dimensionZ']))
        voxelDataset.attrs.create('scale', 1)
        
        metadataGroup = zeroGroup.create_group("MetaData")
        metadataGroup.attrs.create('SoftwareVersion', str(reconRow['softwareVersion']))
        
        pn0DataString = str(reconRow['acquisitionPnOs'])
        if pn0DataString == 'None':
            pn0DataString = '0,0,0,0,0,0'   
        pn0DataListOfString = pn0DataString.split(",")
        pn0DataInverse = np.array(pn0DataListOfString, dtype=np.float32)
        pn0Data = np.reshape(pn0DataInverse,(-1,1))
        pn0Dataset = metadataGroup.create_dataset('pN0Seeds', pn0Data.shape, dtype = float)
        pn0Dataset[:] = pn0Data
        
        l1OverNData = str(reconRow['l1OverN'])
        kVpData = str(reconRow['kVp'])
        mAData = str(reconRow['mA'])
        mAsData = str(reconRow['mAs'])
        metadataGroup.attrs.create('l1OverN', l1OverNData)
        metadataGroup.attrs.create('kVp',kVpData )
        metadataGroup.attrs.create('mA', mAData)
        metadataGroup.attrs.create('mAs', mAsData)
        
        if int(reconRow['acquisitionType']) == 0:
            acquisitionTypeData = 'Fluro'
        elif int(reconRow['acquisitionType']) == 1:
            acquisitionTypeData = 'Single'
        
        metadataGroup.attrs.create('Acquisition Type', acquisitionTypeData)
        
        if int(reconRow['isDual']) == 0:
            dualTypeData = 'No'
        elif int(reconRow['isDual']) == 1:
            dualTypeData = 'Yes'
        
        metadataGroup.attrs.create('Is_Dual', dualTypeData)
        
        if int(reconRow['isPlayback']) == 0:
            playbackTypeData = 'No'
        elif int(reconRow['isPlayback']) == 1:
            playbackTypeData = 'Yes'
            
        metadataGroup.attrs.create('Is_Playback', playbackTypeData)
        
        patientDataGroup = hdf5File.create_group("Patient_Data")
        patientDataGroup.attrs.create('SiteID', str(procedureRow['siteID']))
        patientDataGroup.attrs.create('Total_Dose', str(procedureRow['mGy']))
        patientDataGroup.attrs.create('Total_Dose_Area_Product', str(procedureRow['DAP']))
        patientDataGroup.attrs.create('Distance_To_Reference_Point', str(procedureRow['distanceToReferencePoint']))
        patientDataGroup.attrs.create('Total_Number_Of_Radiographic_Frames', str(procedureRow['totalNumberOfRadiographicFrames']))
        patientDataGroup.attrs.create('Total_Fluoro_Time', str(procedureRow['xRayMins']))
        
        
def main(argv):
    host_ = HOST
    user_ = USER
    password_ = PASSWORD
    database_ = DATABASE
    RECONID = ''
    fileName = ''
    
    try:
        opts, args = getopt.getopt(argv,"hs:u:p:d:f:r:",["hostName =","username =", "password =", "databaseName =", "fileName(specify full path along with full name) = ", "reconID"])
    except getopt.GetoptError:
        print ('createHDF5.py -s <hostName> -u <username> -p <password> -d <databaseName> -f <fileName(specify full path along with full name)> -r <reconID>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print ('createHDF5.py -s <hostName> -u <username> -p <password> -d <databaseName> -f <fileName(specify full path along with full name)> -r <reconID>')
            sys.exit()
        elif opt in ("-s", "--hostName"):
            host_ = arg
        elif opt in ("-u", "--username"):
            user_ = arg
        elif opt in ("-p", "--password"):
            password_ = arg
        elif opt in ("-d", "--databaseName"):
            database_ = arg
        elif opt in ("-f", "--fileName"):
            fileName = arg
        elif opt in ("-r", "--reconID"):
            RECONID = arg
    createHDF5Object = CreateHDF5(HOST, USER, PASSWORD, DATABASE, RECONID)
    reconRow, procedureRow = createHDF5Object.fetchDataFromDatabase()
    createHDF5Object.createHDF5(fileName, reconRow, procedureRow)        

if __name__ == "__main__":
   main(sys.argv[1:])         
    




