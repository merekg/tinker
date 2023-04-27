import numpy as np 
import csv

dbFilePath = '/media/save/Shalin/MLVERTEBRAELABELLING/Files/verifiedVertebraeLabel.csv'
labelFilePath = '/media/save/Shalin/MLVERTEBRAELABELLING/Labels/'
# labelsDB = np.genfromtxt(, delimiter = ',')

with open(dbFilePath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\n')
    for row in csv_reader:
    	g = row[0].split(',[[')
    	prefix = '[['
    	points = g[1]
    	points = prefix + points
    	b = points.split('], [')
    	b[0] = b[0][2:]
    	b[-1] = b[-1][:len(b[-1]) -2]
    	d = []
    	for a in b:
    		d.append(a.split(', '))
    	c = np.asarray(d)
    	c = c.astype('float')
    	print(g[0], c.shape)
    	np.save(labelFilePath + g[0] + '.npy', c)