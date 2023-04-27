#!/usr/bin/python -B
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import nrrd
import header
import inoutCal
import visual
from cStringIO import StringIO

__author__ = 'Arvi Cheryauka'

plt.close("all")

# Verify VISUALLY calibration results produced by C calibration
# Issue: PY changes the order when it write / read, ie swap indexes in 2D array

# ----------------------------------------------------------------------------
# Projects a point from 3D space given the system parameters
# Uses compute_pm() that describes the fixed panel tomosynthesis geometry
# Returns the projection of the 3D point


def project_point(PM, point):
    pt = np.matrix([1.0, 1.0, 1.0, 1.0]).T
    pt[0] = point[0]
    pt[1] = point[1]
    pt[2] = point[2]
    projection = PM * pt
    projection /= projection[2]
    return projection

# ----------------------------------------------------------------------------
#  MAIN PROGRAMM

# I / O
phanFile, imageFile, calibFile, recovFile = inoutCal.main(sys.argv[1:])

# If needed: Create phantom file and conf file's links in yourWorkingDirectory
# os.system("ln -s ~/calibration/calibrationCPP/" + phanFile)
# os.system("ln -s ~/calibration/calibrationCPP/configuration.ini")

# Read phantom data
dataPhan = open(phanFile, 'r').read()
dataXYZ = np.genfromtxt(StringIO(dataPhan.replace('ball:', '')), delimiter=',',
                        usecols=(0, 1, 2), dtype='float', skip_header=2)
nBB, m = dataXYZ.shape

# Read metadata & image data
imgFolder = os.path.dirname(imageFile)
baseIn= os.path.basename( imageFile )
nameIn= os.path.splitext(baseIn)[0]

encoding, sizes, datafile, endian, dtype, dimension = header.read(imageFile)
nu, nv, nView = sizes
nMin, nMax = np.genfromtxt(StringIO(datafile.replace('./'+nameIn+'.%d.raw', '')),
                           usecols=(0, 1), dtype='int')

# Read calibration metadata,angle, PMs
dataCal = open(calibFile, 'r').read()
du_mm, dv_mm = np.genfromtxt(StringIO(dataCal.replace('header:', '')), delimiter=',',
                             usecols=(0, 1), skip_header=4, skip_footer=nView)
data = np.genfromtxt(StringIO(dataCal.replace(':', ',')),
                     delimiter=',', skip_header=6,
                     usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
angle = data[:, 0]
dataPM = data[:, 1:13]

# Read CentroidRecovered data
centRec = np.loadtxt(recovFile, delimiter=',', dtype='float32')

# Loop over images
# plt.figure(figsize=(10,10))
#plt.get_current_fig_manager().window.setGeometry(1800, 600, 700, 700)
UV_CompP = np.zeros((nBB, 2), dtype='float32', order='F')
errRMS = np.zeros((nView, 1), dtype='float32', order='F')
errMAX = np.zeros((nView, 1), dtype='float32', order='F')
for i in range(0, nView):
    # Read image data
    dataImg, options = nrrd.read(imgFolder + '/' +nameIn +'.' + str(i+nMin) + '.nhdr')
    dataImg = dataImg.T
    dataImg = dataImg.astype(np.float)

    visual.image(dataImg, angle[i], 'gray', 1, 8000)

# Segmented centroids from C calibration
    plt.scatter(centRec[2*i, :], centRec[2*i+1, :], color='y', marker='.', s=100)

# PM from C calibration
    PM_compC = dataPM[i, :]
    PM_compC = np.reshape(PM_compC, (4, 3)).T

# Re-project BBs
    for j in range(0, nBB):
        point = dataXYZ[j, :]
        uv = project_point(PM_compC, point)
        UV_CompP[j, 0] = uv[0]
        UV_CompP[j, 1] = uv[1]

# Reprojected using C-calibrate PM and Phantom
    plt.scatter(UV_CompP[:, 0], UV_CompP[:, 1], color='m', marker='.', s=10)

# RMS
    dist_u = (centRec[2*i, :] - UV_CompP[:, 0].T)*du_mm
    dist_v = (centRec[2*i+1, :] - UV_CompP[:, 1].T)*dv_mm
    e = np.sqrt(dist_u**2 + dist_v**2)
    errRMS[i] = e.mean()
    errMAX[i] = e.max()
    plt.text(375, -50, 'RMS / MAX errors = ' +
             "{:.3f} / {:.3f}".format(e.mean(), e.max()) + ' mm',
             color='m')

    plt.pause(0.01)
    if i< nView-1: plt.clf()

# Last frame hold
plt.waitforbuttonpress(); plt.clf()

# Display overall errors
#plt.get_current_fig_manager().window.setGeometry(2000, 1, 500, 500)
visual.error(errRMS, errMAX, angle)
plt.waitforbuttonpress()

print('Overall RMS / MAX errors = ' +
      "{:.3f} / {:.3f}".format(errRMS.mean(), errMAX.max()) + ' mm')
#plt.show()