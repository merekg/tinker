import sys, getopt, os
import pymysql
import numpy as np 
np.set_printoptions(suppress=True)
#Global variables

#Database connection variables
HOST = 'localhost'
USER = 'root'
PASSWORD = ''
DATABASE = 'nView'

class WriteToDatabase:
	def __init__(self, HOST, USER, PASSWORD, DATABASE):
		self.conn = pymysql.connect(host = HOST, user = USER, password = PASSWORD, database = DATABASE)

	def writeToDatabase(self, results):
		cursor = self.conn.cursor()
		for i in range(0, results.shape[0]):
			row = results[i]
			parameters = row[:-1]
			parameterString = ','.join(str(e) for e in parameters)
			query  = 'update ai set smartScrollParameters = \'' + parameterString + '\' where aiID = ' + str(int(row[-1]))
			cursor.execute(query)
			self.conn.commit()


	def readResults(self, filepath):
		results = np.genfromtxt(filepath, delimiter = ',')
		return results



def main(argv):
	host_ = HOST
	user_ = USER
	password_ = PASSWORD
	database_ = DATABASE
	filePath = '/home/tristanmary/rt3d/ML/SmartScrolling/predictions_full.csv'

	getParameters = WriteToDatabase(host_, user_, password_, database_)
	results = getParameters.readResults(filePath)
	getParameters.writeToDatabase(results)
	# results = getParameters.getRowFromTable()
	# getParameters.generateParameters(results)


if __name__ == "__main__":
   main(sys.argv[1:])