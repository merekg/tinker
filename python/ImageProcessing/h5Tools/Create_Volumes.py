#!/usr/bin/python3
import numpy as np 
import h5py
import shutil
from scipy.ndimage import gaussian_filter
import random
import os
import math
import sys, getopt


scale = 10922.5


def create_object(x_hat, y_hat, z_hat, a,b,c, theta, phi, psi, mu, o ):
	
	E = {}
	E['x_hat'] = x_hat 
	E['y_hat'] =  y_hat
	E['z_hat'] = z_hat
	E['a'] = a		
	E['b'] = b 		
	E['c'] = c 		
	E['theta'] = (theta * math.pi) /180
	E['phi'] = (phi * math.pi) /180
	E['psi'] = (psi * math.pi) /180
	E['mu'] = mu
	if o =='water' or o== 'air':
		E['object'] = 'ellipsoid'
	if o == 'bone':
		E['object'] = 'cylinder'
	if o == 'metal':
		E['object'] = 'cylinder'

	return E
	


def create_volumes(projections,vol_extent, resolution, E):
	x_extent = vol_extent[0]
	y_extent = vol_extent[1]
	z_extent = vol_extent[2]

	t = np.linspace(-(x_extent/2), (x_extent/2), resolution)
	xc, yc  =np.meshgrid(t,t)
	zc = np.linspace(0, z_extent, resolution)
	
	if E['object'] == 'ellipsoid':
		
		x_hat = E['x_hat']
		y_hat = E['y_hat']
		z_hat = E['z_hat']
		a = E['a']
		b = E['b']
		c = E['c']
		theta = E['theta']
		phi = E['phi']
		psi = E['psi']
		mu = E['mu']
		d = np.array([1/a**2, 1/b**2, 1/c**2])
		A = np.diag(d)
		n1 = np.array([[math.cos(theta)* math.cos(phi), math.cos(theta)*math.sin(phi), -math.sin(theta)]])
		n2 = np.array([[-math.sin(phi), math.cos(phi), 0]])
		e1 = math.cos(psi) * n1 + math.sin(psi) * n2
		e2 = -math.sin(psi) * n1 + math.cos(psi) * n2
		e3 = np.array([[math.cos(phi)*math.sin(theta), math.sin(phi)*math.sin(theta), math.cos(theta)]])
		Q = np.concatenate((e1,e2,e3))
		E_E_1 = np.matmul(np.transpose(Q),A)
		E_E = np.matmul(E_E_1, Q)
		
		
		equation1_part1 = E_E[0,0]*((xc - x_hat)**2) + 2*(E_E[0,1]* ((xc- x_hat) * (yc-y_hat ))) + E_E[1,1]*((yc - y_hat)**2) 
		for i in range(0, resolution):
			z = zc[i]
			equation = equation1_part1 + 2*(E_E[1,2]* ((xc- x_hat) * (z-z_hat ))) + E_E[2,2]*((z - z_hat)**2) + 2*(E_E[1,2]* ((yc- y_hat) * (z-z_hat ))) 
			s = projections[:,:,i]
			s[equation < 1] = mu * scale
			projections[:,:,i] = s
		
	

	if E['object'] == 'cylinder':
		x_hat = E['x_hat']
		y_hat = E['y_hat']
		z_hat = E['z_hat']
		a = E['a']
		b = E['b']
		c = E['c']
		theta = E['theta']
		phi = E['phi']
		psi = E['psi']
		mu = E['mu']
		d = np.array([0, 1/b**2, 1/c**2])
		A = np.diag(d)
		n1 = np.array([[math.cos(theta)* math.cos(phi), math.cos(theta)*math.sin(phi), -math.sin(theta)]])
		n2 = np.array([[-math.sin(phi), math.cos(phi), 0]])
		e1 = math.cos(psi) * n1 + math.sin(psi) * n2
		e2 = -math.sin(psi) * n1 + math.cos(psi) * n2
		e3 = np.array([[math.cos(phi)*math.sin(theta), math.sin(phi)*math.sin(theta), math.cos(theta)]])
		Q = np.concatenate((e1,e2,e3))
		E_E_1 = np.matmul(np.transpose(Q),A)
		E_E = np.matmul(E_E_1, Q)
		

		equation1_part1 = E_E[0,0]*((xc - x_hat)**2) + 2*(E_E[0,1]* ((xc- x_hat) * (yc-y_hat ))) + E_E[1,1]*((yc - y_hat)**2)
		equation2_part1 = ((xc - x_hat)*e1[0,0]) + ((yc - y_hat)*e1[0,1])
		for i in range(0, resolution):
			z = zc[i]
			equation = equation1_part1 + 2*(E_E[1,2]* ((xc- x_hat) * (z-z_hat ))) + E_E[2,2]*((z - z_hat)**2) + 2*(E_E[1,2]* ((yc- y_hat) * (z-z_hat ))) 
			equation_2 = equation2_part1 + ((z - z_hat)*e1[0,2])
			s = projections[:,:,i]
			indices = np.logical_and(equation < 1, np.absolute(equation_2) < a)
			s[indices] = mu * scale
			projections[:,:,i] = s
		
	return projections


def create_big_water():
	c = random.randint(50, 200)
	x_hat = random.randint(-50, 50)
	y_hat = random.randint(-50, 50)
	z_hat = random.randint(50, 150)
	a =random.randint(100, 200) 
	b = random.randint(200, 250)
	
	mu = random.uniform(0.09,0.14)
	theta = 0
	phi = 0
	psi = 0
	o = 'water'
	E= create_object(x_hat, y_hat, z_hat, a,b,c, theta, phi, psi, mu, o )
	return E


def lung():
	c_water = random.randint(50, 200)
	x_hat_water = random.randint(-50, 50)
	y_hat_water = random.randint(-50, 50)
	z_hat_water = random.randint(50, 150)
	a_water =random.randint(100, 200) 
	b_water = random.randint(200, 250)
	
	mu = random.uniform(0.09,0.14)
	theta = 0
	phi = 0
	psi = 0
	o = 'water'
	E_water= create_object(x_hat_water, y_hat_water, z_hat_water, a_water,b_water,c_water, theta, phi, psi, mu, o )

	x_hat_air1 = x_hat_water+(b_water/4)
	y_hat_air1 = y_hat_water+(a_water/4)
	z_hat_air1 = z_hat_water
	a_air1 = a_water/4
	b_air1 = (b_water/3)
	c_air1 = c_water/3
	mu = 0
	theta = 0
	phi = 0
	psi = 0
	o = 'air'

	E_air1= create_object(x_hat_air1, y_hat_air1, z_hat_air1, a_air1,b_air1,c_air1, theta, phi, psi, mu, o )

	x_hat_air2 = x_hat_water-(b_water/4)
	y_hat_air2 = y_hat_water+(a_water/4)
	z_hat_air2 = z_hat_water
	a_air2 = a_water/4
	b_air2 = (b_water/3)
	c_air2 = c_water/3
	mu = 0
	theta = 0
	phi = 0
	psi = 0
	o = 'air'

	E_air2= create_object(x_hat_air2, y_hat_air2, z_hat_air2, a_air2,b_air2,c_air2, theta, phi, psi, mu, o )



	return E_water, E_air1, E_air2


def create_small_water():
	x_hat = random.randint(-150, 150)
	y_hat = random.randint(-150, 150)
	z_hat = random.randint(1, 250)
	a = random.randint(1, 50)
	b = random.randint(1, 50)
	c = random.randint(1, 50)
	mu = random.uniform(0.09,0.14)
	theta = random.randint(0,90)
	phi = random.randint(-180,180)
	theta = 0
	phi = 0
	psi = 0
	o = 'water'
	E= create_object(x_hat, y_hat, z_hat, a,b,c, theta, phi, psi, mu, o )
	return E

def create_small_air():
	x_hat = random.randint(-150, 150)
	y_hat = random.randint(-150, 150)
	z_hat = random.randint(1, 250)
	a = random.randint(1, 30)
	b = random.randint(1, 30)
	c = random.randint(1, 30)
	mu = 0
	theta = random.randint(0,90)
	phi = random.randint(-180,180)
	theta = 0
	phi = 0
	psi = 0
	o = 'air'

	E= create_object(x_hat, y_hat, z_hat, a,b,c, theta, phi, psi, mu, o )
	return E



def create_bone():
	x_hat_bone = random.randint(-100,100)
	y_hat_bone = random.randint(-100,100)
	z_hat_bone = random.randint(20,180)
	a_bone = random.randint(15, 30)
	b_bone = random.randint(20, 40)
	c_bone = random.randint(20, 40)
	mu_bone = random.uniform(0.14,0.19)
	theta_bone = 0 
	phi_bone = 90
	psi_bone = 0
	o_bone = 'bone'

	E_bone= create_object(x_hat_bone, y_hat_bone, z_hat_bone, a_bone,b_bone,c_bone, theta_bone, phi_bone, psi_bone, mu_bone, o_bone)

	x_hat_water = x_hat_bone
	y_hat_water = y_hat_bone
	z_hat_water = z_hat_bone
	a_water = a_bone	
	b_water = b_bone -5
	c_water = c_bone - 5
	mu_water = random.uniform(0.15,0.24)
	theta_water = theta_bone
	phi_water = phi_bone 
	psi_water = psi_bone
	o_water = 'bone'

	E_water= create_object(x_hat_water, y_hat_water, z_hat_water, a_water,b_water,c_water, theta_water, phi_water, psi_water, mu_water, o_water)

	return E_bone, E_water

def create_metal():
	x_hat = random.randint(-100, 100)
	y_hat = random.randint(-100, 100)
	z_hat = random.randint(20, 180)
	a = random.randint(40, 50)
	b = random.randint(4, 13)
	c = random.randint(4, 13)
	mu = random.uniform(0.35,0.45)
	theta = 0
	phi = 0
	psi = 0
	o = 'metal'

	E= create_object(x_hat, y_hat, z_hat, a,b,c, theta, phi, psi, mu, o )
	return E


def create_scans(num, resolution, output_dir):


	for i in range(1,int(num)+1):
			
		extent = [500,500,400]
		resolution = int(resolution)
		directions = np.zeros((3,3))
		directions[0,0] = 1.0
		directions[1,1] = 1.0
		directions[2,2] = 1.0

		origin = [0.0,0.0,0.0]
		spacing = [extent[0]/resolution, extent[1]/resolution, extent[2]/resolution]


		projections = np.zeros((int(resolution), int(resolution), int(resolution))).astype('uint16')
		
		if i <= 9:
			output_scan_file = output_dir + 'GTVol_000' + str(i) +'.h5'
		if i > 9 and i<=99:
			output_scan_file = output_dir + 'GTVol_00' + str(i) +'.h5'

		if i > 99 and i<= 999:
			output_scan_file = output_dir + 'GTVol_0' + str(i) +'.h5'

		if i > 999:
			output_scan_file = output_dir + 'GTVol_' + str(i) +'.h5'
		f = h5py.File(output_scan_file, 'w')
		ITKImage = f.create_group("ITKImage")
		zero_grp = ITKImage.create_group("0")
		meta_data = zero_grp.create_group("MetaData")
		zero_grp.create_dataset('Dimension', data=np.array([float(resolution), float(resolution), float(resolution)]))
		zero_grp.create_dataset('Directions', data=np.array(directions))
		zero_grp.create_dataset('Origin',data=np.array(origin))
		zero_grp.create_dataset('Spacing', data=np.array(spacing))
		voxelData = zero_grp.create_dataset('VoxelData', data=projections)
		voxelData.attrs['nImages'] = resolution
		voxelData.attrs['scale'] = '10922.5'


		prob_water = random.randint(1,100)
		# if prob_water < 60 and prob_water> 30:
		# 	E_water = create_big_water()
		# 	projections = create_volumes(projections, extent, resolution,E_water)
		if prob_water < 50:
			E_water, E_air1, E_air2 = lung()
			projections = create_volumes(projections, extent, resolution,E_water)
			# projections = create_volumes(projections, extent, resolution,E_air1)
			# projections = create_volumes(projections, extent, resolution,E_air2)


		if prob_water > 50 and prob_water < 70:
			no_small_water = random.randint(5, 10)
			for i in range(0, no_small_water):
				E = create_small_water()
				projections = create_volumes(projections, extent, resolution,E)


		# no_small_air = random.randint(3,7)
		# for i in range(0, no_small_air):
		# 	E = create_small_air()
		# 	projections = create_volumes(projections, extent, resolution,E)

		no_bone = random.randint(10, 25)
		for i in range(0, no_bone):
			E_bone, E_water = create_bone()
			projections = create_volumes(projections, extent, resolution,E_bone)
			projections = create_volumes(projections, extent, resolution,E_water)
		




		prob_metal = random.randint(1,100)
		if prob_metal <= 7:
			no_metal = random.randint(1,2)
			for i in range(no_metal):
				E = create_metal()
				projections = create_volumes(projections, extent, resolution,E)
		if prob_water < 35:
			projections = create_volumes(projections, extent, resolution,E_air1)
			projections = create_volumes(projections, extent, resolution,E_air2)


		projections = np.swapaxes(projections, 0, 2)
		blurr =  random.randint(0,2)
		if blurr == 0:
			projections = projections

		elif blurr == 1:
			projections = gaussian_filter(projections, sigma=0.5)
			

		elif blurr == 2:
			projections = gaussian_filter(projections, sigma=0.25)
			

		
		projections = projections.astype('uint16')
		outputFile = h5py.File(output_scan_file, 'r+')
		voxelData = outputFile['ITKImage/0/VoxelData']
		voxelData[...] = projections
	print('DONE GENERATING OBJECTS')


def main(argv):
	output_scan_file = ''
	num = 1
	resolution = 32
	try:
		opts, args = getopt.getopt(argv,"ho:n:r:",["output_dir=","num_of_scans=", "resolution="])
	except getopt.GetoptError:
		print ('Create_Volumes.py -o <outputfile> -n <num_of_scans> -r <resolution>')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ('-h', '--help'):
			print ('Create_Volumes.py -o <outputfile> -n <num_of_scans> -r <resolution>')
			sys.exit()
		elif opt in ("-o", "--output_dir"):
			output_scan_file = arg
		elif opt in ("-n", "--num_of_scans"):
			num = arg
		elif opt in ("-r", "--resolution"):
			resolution = arg
	create_scans(num, resolution, output_scan_file)


if __name__ == "__main__":
   main(sys.argv[1:])