import pymysql
import pandas
import sys, getopt, os


#Global variables

#Database connection variables
HOST = 'localhost'
USER = 'root'
PASSWORD = ''
DATABASE = 'nView'

#Tables that do not contain sensitive information.
TABLE_NAMES = ['reconstructions', 'screenshots', 'evalformquestions', 'evaluationform']

class ExportDatabase:
    def __init__(self, HOST, USER, PASSWORD, DATABASE):
        self.conn = pymysql.connect(host = HOST, user = USER, password = PASSWORD, database = DATABASE)
        #Default value in the excluded columns 
        self.defaultNumericalValue = 0
        self.defaultTextValue = 'REDACTED'
    
    def getProceduresTable(self, pathToDirectory):
        
        #Columns to exlude
        textColumnsToExclude = ['patientName']
        numericalColumnsToExclude = ['patientMRN', 'patientMonthOfBirth', 'patientDayOfBirth']
        
        procedureCursor = self.conn.cursor()
        procedureQuery = 'select * from procedures'
        procedureResults = pandas.read_sql_query(procedureQuery, self.conn)
        
        #Redact sensitive information
        for column in numericalColumnsToExclude:
            procedureResults[column].values[:] = self.defaultNumericalValue
        
        for column in textColumnsToExclude:
            procedureResults[column].values[:] = self.defaultTextValue
            
        procedureResults.to_csv(pathToDirectory + "procedures.csv", index=False)
    
    def getTable(self, pathToDirectory, tableName):
        
        cursor = self.conn.cursor()
        try:
            query = 'select * from ' + tableName
            results = pandas.read_sql_query(query, self.conn)
            if tableName == 'reconstructions':
                results = results[pandas.isnull(results['acquisitionDate']) == False]
                results['headLeft'].values[:] = self.defaultNumericalValue
                results['backPosition'].values[:] = self.defaultNumericalValue
            
            results.to_csv(pathToDirectory + tableName + ".csv", index=False)
        except pandas.io.sql.DatabaseError as e:
            print(tableName + ' does not exist in the database.')
        


def main(argv):
    host_ = HOST
    user_ = USER
    password_ = PASSWORD
    database_ = DATABASE
    pathToDirectory_ = ''
    
    try:
        opts, args = getopt.getopt(argv,"hs:u:p:d:f:",["hostName =","username =", "password =", "databaseName =", "folderName ="])
    except getopt.GetoptError:
        print ('exportDatabaseToCSV.py -s <hostName> -u <username> -p <password> -d <databaseName> -f <folderName>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print ('exportDatabaseToCSV.py -s <hostName> -u <username> -p <password> -d <databaseName> -f <folderName>')
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
                os.makedirs(pathToDirectory_)
            
                
            
    exportObject = ExportDatabase(host_, user_, password_, database_)
    exportObject.getProceduresTable(pathToDirectory_)
    
    for tableName in TABLE_NAMES:
        exportObject.getTable(pathToDirectory_, tableName)
    
if __name__ == "__main__":
   main(sys.argv[1:])
