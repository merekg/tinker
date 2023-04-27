from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import optimize

x = [47.33027267456055, 76.14234161376953, 106.6542358398437, 134.56455993652344, 161.77902221679688, 183.6833648681640, 201.27322387695312, 221.18626403808594, 245.1279754638672, 267.451904296875]
y = [138.96987915039062, 118.9944152832031, 104.49607086181641, 113.96648406982422, 128.71507263183594, 151.5083770751953, 174.30166625976562, 186.36871337890625, 189.3854675292968, 187.039108276367]
z = [166.2730712890625, 156.556472778320, 153.41474914550782, 147.1931610107422, 149.3897857666015, 151.1576538085937, 152.57723999023438, 154.18438720703125, 148.3625335693359, 142.934005737304]


def test_func(x, a,b,c,d,e,f,g):
    return (a * np.exp(-((x-b) **2)/ (2 *(c ** 2))) - d * np.exp(-((x-e) **2)/ (2 *(f ** 2))) + g)



def trigonometricFunction(x,y,z):
	x_cubic = []
	y_cubic = []
	z_cubic = []
	params_y, params_covariance_y = optimize.curve_fit(test_func, x, y)
	params_z, params_covariance_z = optimize.curve_fit(test_func, x, z)
	print(params_y, params_z)
	for i in range(0, 300):
		x_cubic.append(i)
		y_cubic.append(test_func(i, params_y[0], params_y[1], params_y[2], params_y[3], params_y[4], params_y[5], params_y[6]))
		z_cubic.append(test_func(i, params_z[0], params_z[1], params_z[2], params_z[3], params_z[4], params_z[5], params_z[6]))
	
	fig, (ax1, ax2) = plt.subplots(2)
	ax1.scatter(x, y)
	ax1.plot(x_cubic, y_cubic, '--', color='red')
	ax1.set(xlim=(0, 300), ylim=(min(y) - 5, max(y) + 5))

	ax2.scatter(x, z)
	ax2.plot(x_cubic, z_cubic, '--', color='red')
	ax2.set(xlim=(0, 300), ylim=(min(z) - 5, max(z) + 5))
	fig.savefig('/media/save/Shalin/SmartScrolling/CurveFittingFigures/Figure_Trigonometric.png')


def polynomialPlot(x,y,z, degree):
	x_cubic = []
	y_cubic = []
	z_cubic = []
	a = np.polyfit(x,y,degree)
	b = np.polyfit(x,z, degree)
	print(len(a))
	for i in range (0, 300):
		x_cubic.append(i)
		y_calc = 0
		z_calc = 0
		for d in range(0, degree+1):
			y_calc += a[d] * (i ** (degree-d))
			z_calc += b[d] * (i ** (degree-d))
		y_cubic.append(y_calc)
		z_cubic.append(z_calc)
	fig, (ax1, ax2) = plt.subplots(2)
	ax1.scatter(x, y)
	ax1.plot(x_cubic, y_cubic, '--', color='red')
	ax1.set(xlim=(0, 300), ylim=(min(y) - 5, max(y) + 5))

	ax2.scatter(x, z)
	ax2.plot(x_cubic, z_cubic, '--', color='red')
	ax2.set(xlim=(0, 300), ylim=(min(z) - 5, max(z) + 5))
	fig.savefig('/media/save/Shalin/SmartScrolling/CurveFittingFigures/Figure_Polynomial_Degree'+ str(degree)+ '.png')

# for i in range(2, len(x) - 1):
# 	polynomialPlot(x,y,z,i)

trigonometricFunction(x,y,z)
# smartScrollParameters = [-1.28507996e-05,  6.46455904e-03, -1.03168863e+00,  2.02747034e+02, -4.98263247e-05,  2.63215751e-02, -3.77274931e+00,  2.67909866e+02]

# smartScrollParameters = [0.000012,-0.005236,0.544234,148.647867,-0.000071,0.035036,-4.632666,267.976798]
# x_cubic = []
# y_cubic = []
# z_cubic = []

# for i in range (0, 300):
# 	x_cubic.append(i)
# 	y_cubic.append((smartScrollParameters[4] * (i ** 3)) + (smartScrollParameters[5] * (i ** 2)) + (smartScrollParameters[6] * (i ** 1)) + (smartScrollParameters[7] * (i ** 0)))
# 	z_cubic.append((smartScrollParameters[0] * (i ** 3)) + (smartScrollParameters[1] * (i ** 2)) + (smartScrollParameters[2] * (i ** 1)) + (smartScrollParameters[3] * (i ** 0)))

# fig, (ax1, ax2) = plt.subplots(2)
# ax1.scatter(x, y)
# ax1.plot(x_cubic, y_cubic, '--', color='red')

# ax2.scatter(x, z)
# ax2.plot(x_cubic, z_cubic, '--', color='red')

# plt.show()