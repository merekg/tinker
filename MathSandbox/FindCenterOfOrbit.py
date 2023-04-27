from scipy.spatial.transform import Rotation as R
import sys
import os
import numpy as np

ORIGIN = np.array([55,-38,12.138])

#	fit a sphere to X,Y, and Z data points
#	returns the radius and center points of
#	the best fit sphere
def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2]

def invertTx(Tx):
    rotation = np.array(Tx[:3,:3])
    translation = np.array(Tx[:3,3])
    Tx_i = np.append(rotation.T, -1 *np.matmul(rotation.T, [[translation[0]],[translation[1]],[translation[2]]]), axis=1)
    return np.append(Tx_i, [[0,0,0,1]], axis=0)

#Port Frame	Q0	Qx	Qy	Qz	Tx	Ty	Tz	Error
def parseLine(line):
    ar = line.split(",")
    ar[0] = int(ar[0][5])
    ar[1] = int(ar[1])
    ar[2] = float(ar[2])
    ar[3] = float(ar[3])
    ar[4] = float(ar[4])
    ar[5] = float(ar[5])
    ar[6] = float(ar[6])
    ar[7] = float(ar[7])
    ar[8] = float(ar[8])
    ar[9] = float(ar[9])
    return ar

def createTx(ar):
    qs = ar[0]
    qx = ar[1]
    qy = ar[2]
    qz = ar[3]
    tx = ar[4]
    ty = ar[5]
    tz = ar[6]

    rot = np.array(R.from_quat([qx,qy,qz,qs]).as_matrix())
    Tx = np.append(rot.T, np.array([[tx],[ty],[tz]]), axis=1)
    return np.append(Tx, [[0,0,0,1]], axis=0)

# python tooltip.py touchOutput.csv
def main():
    touch = open(sys.argv[1], 'r')

    # METHOD 1: Just use the touch, average across data
    touchData = []
    for line in touch:
        touchData.append(parseLine(line))

    probe = []
    patient = []
    for data in touchData:
        if data[0] is 1:
            probe.append(data[2:])
        else:
            patient.append(data[2:])
    probe = np.array(probe)

    # Calculate the difference between the dip and the tip
    _tipToDip = np.zeros(3)
    for i in range(len(probe)):
        probeString = probe[i]
        patientString = patient[i]
        oldOrigin = np.array([patientString[4], patientString[5], patientString[6]])

        r_patient = np.array(R.from_quat([patientString[1], patientString[2], patientString[3], patientString[0]]).as_matrix())
        r_probe = np.array(R.from_quat([probeString[1], probeString[2], probeString[3], probeString[0]]).as_matrix())
        dip = np.matmul(r_patient, ORIGIN) + oldOrigin

        tip = np.array([probeString[4], probeString[5], probeString[6]], dtype=float)
        tipToDip = np.matmul(r_probe.T, dip-tip) 

        # running average
        _tipToDip = ( _tipToDip*(i) + tipToDip ) / (i+1)

    print(_tipToDip)

if __name__ == "__main__":
    main()
