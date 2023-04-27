import numpy as np
import sys
import os

def computeRigidTransform(A, B):

    # Verify that: the shape of A and B match, each have 3 rows
    assert A.shape == B.shape
    assert A.shape[0] == 3
    assert B.shape[0] == 3

    # Compute the centroid of each matrix
    Ac = np.mean(A, axis=1)[:,None]
    Bc = np.mean(B, axis=1)[:,None]

    # Compute covariance of centered matricies and find rotation using SVD
    H = np.matmul(A - Ac, (B - Bc).T)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # Correct R for the special reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.matmul(Vt.T, U.T)

    # Compute translation from centroids and rotation matrix
    t = np.matmul(-R, Ac) + Bc

    return R, t

def simulatedTransformation():

    # Simulate all anglular rotations
    theta = np.random.rand() * 2 * np.pi
    phi = np.random.rand() * 2 * np.pi
    psi = np.random.rand() * 2 * np.pi

    # Formulate rotation matrix based on angles
    c, s = np.cos(theta), np.sin(theta)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    c, s = np.cos(phi), np.sin(phi)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    c, s = np.cos(psi), np.sin(psi)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = np.matmul(np.matmul(Rx, Ry), Rz)

    # Simulate a random translation (0 to 10mm)
    t = 10 * np.random.rand(3, 1)

    # Produce a set of N 3D points and compute rotation and translation of points
    N = 100
    Pc = 10 * np.random.rand(3, N)
    Pv = np.matmul(R, Pc) + t

    # Compute rigid transformation from points
    Rc, tc = computeRigidTransform(Pc, Pv)

    # Print out original and computed Rotation and translation for comparison
    print "Original vs. computed rotation and translation"
    print R, '\n', Rc, '\n', t, '\n', tc, '\n'
    print ""

    # Output files for point and rotatin matrix simulation
    print "Saving simulated files to /tmp"
    np.savetxt("/tmp/VolumePoints.txt", Pv.T, delimiter=",")
    print "Simulated volume points saved as \"/tmp/VolumePoints.txt\""
    np.savetxt("/tmp/CameraPoints.txt", Pc.T, delimiter=",")
    print "Simulated camera points saved as \"/tmp/CameraPoints.txt\""
    np.savetxt("/tmp/Rotation.txt", R, delimiter=",")
    print "Simulated rotation matrix saved as \"/tmp/Rotation.txt\""
    np.savetxt("/tmp/Translation.txt", t, delimiter=",")
    print "Simulated translation matrix saved as \"/tmp/Translation.txt\""
    print ""

def toTransformationMatrix(rotationMatrix, translationVector):

    # Start the transformation matrix as the identity matrix
    transformationMatrix = np.identity(4)

    # Fill in the rotation and translation parts of the transformation matrix
    transformationMatrix[0:3,0:3] = rotationMatrix
    transformationMatrix[0:3,3] = translationVector[:,0];

    return transformationMatrix

def findTransformation():

    # Verify that: two arguments ar given, each is a vaild file path
    assert len(sys.argv) == 4
    assert os.path.isfile(sys.argv[1])
    assert os.path.isfile(sys.argv[2])
    assert not os.path.isfile(sys.argv[3])

    # Read in points from files given as commandline arguments
    CameraPoints = np.genfromtxt(sys.argv[1], delimiter=',').T
    VolumePoints = np.genfromtxt(sys.argv[2], delimiter=',').T

    # Compute rigid transform
    rotationMatrix, translationVector = computeRigidTransform(CameraPoints, VolumePoints)

    # Print rotation and translation matricies
    print "Computed rotation and translation matricies"
    print "Rotation:\n", rotationMatrix
    print "Translation:\n", translationVector
    print ""

    # Formulate transformation matrix from rotation and translation matricies
    transformationMatrix = toTransformationMatrix(rotationMatrix, translationVector)

    # Output files for point and rotatin matrix simulation
    print "Saving calibration as", sys.argv[3]
    np.savetxt(sys.argv[3], transformationMatrix , delimiter=" ")
    print ""

if __name__ == "__main__":
    findTransformation()
    #simulatedTransformation()
