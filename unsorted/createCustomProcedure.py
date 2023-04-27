import pymysql
import pandas
import sys, getopt, os
import time
import json
from datetime import datetime

#Global variables

#Database connection variables
HOST = 'localhost'
USER = 'root'
PASSWORD = ''
DATABASE = 'nView'


class CreateCustomProcedure:
    def __init__(self, HOST, USER, PASSWORD, DATABASE):
        #Establish connection
        self.conn = pymysql.connect(host = HOST, user = USER, password = PASSWORD, database = DATABASE, cursorclass=pymysql.cursors.DictCursor)
    
    def updateProcedureTable(self, procedureData, numberOfImages):
        #Force a sleep to prevent duplication of keys
        time.sleep(1)
        databaseCursor = self.conn.cursor()
        ProcedureID = str(int(time.time()))
        currentDate = datetime.now()
        formatted_date = currentDate.strftime('%Y-%m-%d %H:%M:%S')
        #Build the query
        query = 'insert into procedures (ProcedureID, procedureName, physicianName, patientGender, patientName, patientMRN, patientYearOfBirth, siteID, procedureDate, numImages) values ('+ ProcedureID + ', \'' + procedureData['procedureName'] + '\', \'' + procedureData['physicianName'] + '\', \'' + procedureData['patientGender'] + '\', \'' + procedureData['patientName'] + '\', ' + procedureData['patientMRN'] +', ' + procedureData['patientYearOfBirth'] + ', \'' + procedureData['siteID'] + '\', \'' + formatted_date + '\', ' + str(numberOfImages) +' );'
        
        numberOfRowsAffected = databaseCursor.execute(query)
                
        if numberOfRowsAffected == 0:
            self.updateProcedureTable(procedureData)
        # Commit the transaction
        self.conn.commit()
        return ProcedureID
    
    def readTemplate(self, filePath):
        with open(filePath) as json_file: 
            data = json.load(json_file) 
        return data
    
    def updateReconstructionTable(self, ReconID, ProcedureID):
        time.sleep(1)
        databaseCursor = self.conn.cursor()
        newReconID = str(int(time.time()))
        query = 'select * from reconstructions where ReconID = \'' + ReconID + '\';' 
        numberOfRowsAffected = databaseCursor.execute(query)
        result = databaseCursor.fetchall()
        row = result[0]        
        self.conn.commit()
        
        row['ReconID'] = newReconID
        row['ProcedureID'] = ProcedureID
        del row['acquisitionDate'] 
        del row['headLeft']
        del row['backPosition']
        del row['isSpine']
        insertQuery = 'insert into reconstructions ('
        for fieldName in row:
            if row[fieldName] != 'None':
                insertQuery += fieldName + ','
        insertQuery = insertQuery[:-1]
        insertQuery = insertQuery + ') values ('
        for fieldName in row:
            if row[fieldName] != 'None':
                insertQuery += '\'' + str(row[fieldName]) + '\','
        insertQuery = insertQuery[:-1]
        insertQuery = insertQuery + ');'
        numberOfRowsAffected = databaseCursor.execute(insertQuery)          
        
        self.conn.commit()     


def main(argv):
    host_ = HOST
    user_ = USER
    password_ = PASSWORD
    database_ = DATABASE
    
    try:
        opts, args = getopt.getopt(argv,"hs:u:p:d:f:",["hostName =","username =", "password =", "databaseName ="])
    except getopt.GetoptError:
        print ('createCustomProcedure.py -s <hostName> -u <username> -p <password> -d <databaseName>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print ('createCustomProcedure.py -s <hostName> -u <username> -p <password> -d <databaseName>')
            sys.exit()
        elif opt in ("-s", "--hostName"):
            host_ = arg
        elif opt in ("-u", "--username"):
            user_ = arg
        elif opt in ("-p", "--password"):
            password_ = arg
        elif opt in ("-d", "--databaseName"):
            database_ = arg
    
    createCustomProcedureObject = CreateCustomProcedure(host_, user_, password_, database_)
    listOfProcedures = createCustomProcedureObject.readTemplate('customProcedureTemplate.json')
    for procedure in listOfProcedures:
        print('Started creating custom procedure')
        procedureData = listOfProcedures[procedure]
        listOfReconstructions = procedureData[0]['reconIDList']
        ProcedureID = createCustomProcedureObject.updateProcedureTable(procedureData[0], len(listOfReconstructions))
        listOfReconstructions = procedureData[0]['reconIDList']
        for reconID in listOfReconstructions:
            createCustomProcedureObject.updateReconstructionTable(reconID, ProcedureID)
            print('Added an Image')
        print('Created custom procedure')

if __name__ == "__main__":
   main(sys.argv[1:])    
    


