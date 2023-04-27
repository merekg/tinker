import boto3
import os
import sys
import threading
import logging
from botocore.exceptions import ClientError
import pymysql

#Global Variables
BUCKET_NAME = ""

#Database connection variables
HOST = 'localhost'
USER = 'root'
PASSWORD = ''
DATABASE = ''

class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()
    def __call__(self, bytes_amount):
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()

class UpdateDatabase:
    def __init__(self, HOST, USER, PASSWORD, DATABASE):
        self.conn = pymysql.connect(host = HOST, user = USER, password = PASSWORD, database = DATABASE)

    def updateDatabase(self, fileName):
        databaseCursor = self.conn.cursor()
        #Extract the recon ID
        reconID = int(fileName.split('.')[0])
        # Build the query
        query = 'update reconstructions set savePath = \'s3://' + BUCKET_NAME + '/' + fileName + '\' where ReconID = ' + str(reconID) + ';'
        numberOfRowsAffected = databaseCursor.execute(query)

        # Commit the transaction
        self.conn.commit()

        if numberOfRowsAffected == 0:
            print ('\nDid not change the database\n')
        elif numberOfRowsAffected == 1:
            print ('\nRow updated in the database\n')
        
        return numberOfRowsAffected

class UploadToS3:
    def __init__(self):
        self.s3 = boto3.client('s3')

    def uploadFiles(self, pathToDirectory, HOST, USER, PASSWORD, DATABASE):
        for file in os.listdir(pathToDirectory):
            # If S3 object_name was not specified, use file name
            objectName = file
            filePath = pathToDirectory + file
            # Check if the file is a .dat file.
            if filePath.endswith('.dat'):
                try:
                    databaseObject = UpdateDatabase(HOST, USER, PASSWORD, DATABASE)
                    numberOfRowsAffected = databaseObject.updateDatabase(file)
                    if numberOfRowsAffected == 1:
                        response = self.s3.upload_file(filePath, BUCKET_NAME, objectName, Callback=ProgressPercentage(filePath))                    
                    elif numberOfRowsAffected == 0:
                        print('Did not upload file check again'+ filePath)
                except ClientError as e:
                    logging.error(e)
                    print ('Failed to upload file: ' + file)
                print ('\nSuccessfully uploaded file '+ file + ' to S3 bucket\n')
            else:
                print ('Incorrect file type. (Not a .dat file)')

def main(argv):
    host_ = HOST
    user_ = USER
    password_ = PASSWORD
    database_ = DATABASE
    pathToDirectory_ = ''
    
    try:
        opts, args = getopt.getopt(argv,"hs:u:p:d:f:",["hostName =","username =", "password =", "databaseName =", "folderName ="])
    except getopt.GetoptError:
        print ('uploadDatFilesToS3.py -s <hostName> -u <username> -p <password> -d <databaseName> -f <folderName>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print ('uploadDatFilesToS3.py -s <hostName> -u <username> -p <password> -d <databaseName> -f <folderName>')
            sys.exit()
        elif opt in ("-s", "--hostName"):
            host_ = arg
        elif opt in ("-u", "--username"):
            user_ = arg
        elif opt in ("-p", "--password"):
            password_ = arg
        elif opt in ("-d", "--databaseName"):
            database_ = arg
        elif opt in ("-f", "--folderName"):
            pathToDirectory_ = arg
            if not pathToDirectory_.endswith('/'):
                pathToDirectory_ = pathToDirectory_ + '/'
            if not os.path.exists(pathToDirectory_):
                print ('Specified file path does not exist. Please check again and try')

    uploadObject = UploadToS3()
    uploadObject.uploadFiles(pathToDirectory_, host_, user_, password_, database_)
    
if __name__ == "__main__":
   main(sys.argv[1:])  
