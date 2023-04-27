import pandas
from sqlalchemy import create_engine
import pymysql
import sys, getopt, os
#Global variables

#Database connection variables
HOST = 'localhost'
USER = 'root'
#password is empty as we have empty passwords for all our databases.
PASSWORD = ''
DATABASE = 'nViewCentral'

#Tables and their Primary keys.
TABLE_NAMES = {'procedures':['ProcedureID'], 'reconstructions':['ReconID'], 'screenshots':['ScreenshotID'], 'evalformquestions':['questionID'], 'evaluationform':['ProcedureID']}




class ImportTables:
    def __init__(self, HOST, USER, PASSWORD, DATABASE, pathToDirectory, tableName):
        #Create a SQL engine 
        self.engine = create_engine('mysql+pymysql://'+ USER + ':' + PASSWORD + '@' + HOST + '/' + DATABASE + '?charset=utf8mb4')
        #Get the dataframe of the table
        try:
            self.dataframe = pandas.read_csv(pathToDirectory + tableName + '.csv')
        except FileNotFoundError as e:
            print(tableName + ' does not exist in the folder.')
    
    def cleanDuplicates(self, tableName, duplicateColumns=[]):
        #Get the existing column from the central database.
        args = 'SELECT %s FROM %s' %(', '.join(['{0}'.format(col) for col in duplicateColumns]), tableName)
        #drop the duplicates from the dataframe        
        self.dataframe.drop_duplicates(duplicateColumns, keep='last', inplace=True)
        #Merge and drop duplicate rows
        self.dataframe = pandas.merge(self.dataframe, pandas.read_sql(args, self.engine), how='left', on=duplicateColumns, indicator=True)
        self.dataframe = self.dataframe[self.dataframe['_merge'] == 'left_only']
        self.dataframe.drop(['_merge'], axis=1, inplace=True)
   
    def appendTable(self, tableName):
        #Append the dataframe to the table in the database. 
        self.dataframe.to_sql(tableName, self.engine, if_exists='append', index=False)
        
    def checkForDifferentColumns(self, tableName):
        query = 'select * from '+ tableName
        oldDatabase = pandas.read_sql(query, self.engine)
        differentColumns = list(set(self.dataframe) - set(oldDatabase))
        for col in differentColumns:
            del self.dataframe[col]
        
        
        
        
        
def main(argv):
    host_ = HOST
    user_ = USER
    password_ = PASSWORD
    database_ = DATABASE
    pathToDirectory_ = ''
    
    try:
        opts, args = getopt.getopt(argv,"hs:u:p:d:f:",["hostName =","username =", "password =", "databaseName =", "folderName ="])
    except getopt.GetoptError:
        print ('importTableInToCentralDatabase.py -s <hostName> -u <username> -p <password> -d <databaseName> -f <folderName>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print ('importTableInToCentralDatabase.py -s <hostName> -u <username> -p <password> -d <databaseName> -f <folderName>')
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
                print('Specified file path does not exist. Please check again and try')
    
    for tableName, primaryKey in TABLE_NAMES.items():
        if os.path.isfile(pathToDirectory_ + tableName + '.csv'):
            importObject = ImportTables(host_, user_, password_, database_,pathToDirectory_, tableName)
            importObject.cleanDuplicates(tableName, primaryKey)
            importObject.checkForDifferentColumns(tableName)
            importObject.appendTable(tableName)
    
if __name__ == "__main__":
   main(sys.argv[1:])



