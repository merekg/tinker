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

class GetParameters:
	def __init__(self, HOST, USER, PASSWORD, DATABASE):
		self.conn = pymysql.connect(host = HOST, user = USER, password = PASSWORD, database = DATABASE)

	def getRowFromTable(self):
		cursor = self.conn.cursor()
		query = 'select ReconID, startPoint, endPoint from ai'
		cursor.execute(query)
		output = cursor.fetchall()
		return output

	def generateParameters(self, results):
		labelList = []
		for row in results:
			reconID = int(row[0])
			startPoint = [float(idx) for idx in row[1].split(',')]
			endPoint = [float(idx) for idx in row[2].split(',')]
			A = np.array([[startPoint[0], 1, 0, 0], [0,0,startPoint[0], 1], [endPoint[0], 1, 0, 0], [0,0,endPoint[0],1]])
			B = np.array([startPoint[2],startPoint[1],endPoint[2],endPoint[1]])
			X = np.linalg.solve(A,B)
			label = np.append(X, reconID)
			labelList.append(label)
		Y = np.asarray(labelList)
		print(Y)
		np.savetxt("Labels.csv", Y, delimiter=",")

	def generateTrainParametersFromCSV(self, filePath):
		labelList = []
		data = np.genfromtxt(filePath, delimiter = ',')
		for i in range(0, data.shape[0]):
			row = data[i]
			reconID = int(row[1])
			A = np.array([[row[2]**3, row[2]**2, row[2], 1, 0, 0, 0, 0], [0, 0, 0, 0, row[2]**3, row[2]**2, row[2], 1], [row[6]**3, row[6]**2, row[6], 1, 0, 0, 0, 0], [0, 0, 0, 0, row[6]**3, row[6]**2, row[6], 1], [row[10]**3, row[10]**2, row[10], 1, 0, 0, 0, 0], [0, 0, 0, 0, row[10]**3, row[10]**2, row[10], 1], [row[14]**3, row[14]**2, row[14], 1, 0, 0, 0, 0], [0, 0, 0, 0, row[14]**3, row[14]**2, row[14], 1]])
			B = np.array([row[4],row[3],row[8],row[7], row[12], row[11], row[16], row[15]])
			print(A.shape, B.shape, reconID)
			X = np.linalg.solve(A,B)
			label = np.append(X, reconID)
			labelList.append(label)
		Y = np.asarray(labelList)
		print(Y)
		np.savetxt("/media/save/Shalin/SmartScrolling/CubicFiles/augmentedTrainLabelsCubic.csv", Y, delimiter=",", fmt='%f')

	def generateTestParametersFromCSV(self, filePath):
		labelList = []
		data = np.genfromtxt(filePath, delimiter = ',')
		for i in range(0, data.shape[0]):
			row = data[i]
			reconID = int(row[1])
			A = np.array([[row[2]**3, row[2]**2, row[2], 1, 0, 0, 0, 0], [0, 0, 0, 0, row[2]**3, row[2]**2, row[2], 1], [row[6]**3, row[6]**2, row[6], 1, 0, 0, 0, 0], [0, 0, 0, 0, row[6]**3, row[6]**2, row[6], 1], [row[10]**3, row[10]**2, row[10], 1, 0, 0, 0, 0], [0, 0, 0, 0, row[10]**3, row[10]**2, row[10], 1], [row[14]**3, row[14]**2, row[14], 1, 0, 0, 0, 0], [0, 0, 0, 0, row[14]**3, row[14]**2, row[14], 1]])
			B = np.array([row[4],row[3],row[8],row[7], row[12], row[11], row[16], row[15]])
			print(A.shape, B.shape, reconID)
			X = np.linalg.solve(A,B)
			label = np.append(X, reconID)
			labelList.append(label)
		Y = np.asarray(labelList)
		print(Y)
		np.savetxt("testLabelsCubic.csv", Y, delimiter=",", fmt='%f')






def main(argv):
	host_ = HOST
	user_ = USER
	password_ = PASSWORD
	database_ = DATABASE
	filePath = '/media/save/Shalin/SmartScrolling/CubicFiles/augmentedCubicTrainLabel.csv'
	filePathTest = '/media/save/Shalin/SmartScrolling/Files/cubicTestLabel.csv'
	getParameters = GetParameters(host_, user_, password_, database_)
	getParameters.generateTrainParametersFromCSV(filePath)
	# getParameters.generateTestParametersFromCSV(filePathTest)



if __name__ == "__main__":
   main(sys.argv[1:])    	