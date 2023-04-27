import h5py
import sys
from math import isnan

filePath = sys.argv[1]
pose = h5py.File(filePath,'r')['ITKImage/0/MetaData/imagePnO'][:]
gamma = pose[3]
if(isnan(gamma)):
    signGamma = 0
else:
    signGamma = gamma/abs(gamma)
beta = pose[4]
crossArm = float(sys.argv[2])/2.0

metaTable = h5py.File(filePath, 'r+')['ITKImage/0/MetaData/metaDataTable']
metaTable.attrs.create('GammaRotationDegree', data=str(gamma))
metaTable.attrs.create('BetaRotationDegree', data=str(beta))
metaTable.attrs.create('CrossArmTranslationMM', data=str(-signGamma*crossArm))
