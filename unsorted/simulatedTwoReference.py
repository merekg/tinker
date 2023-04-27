import sys
import numpy as np

def rotationMatrixDistance(R):
    return np.degrees(np.arccos((np.trace(R)-1)/2))

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


double_tool_file = np.loadtxt(sys.argv[2], delimiter=",").reshape((3,-1))
large_tool_file = np.array([0,0,0,107.51,-9.41,0,0,115.84,0,140,108.84,0,92.47,297.65,0,-33.14,282.22,0]).reshape((3,-1))
small_tool_file = np.array([ 0,0,0,-29.08,-27.47,0,0,-102,0,-59.1,-139.52,0,3.54,-210.94,0,-70.35,-223.77,0 ]).reshape((3,-1))

diffs = np.loadtxt(sys.argv[1], delimiter=",")

print("small array")
max_angle = -1
for line in diffs:
    rotationMatrix, translationVector = computeRigidTransform(small_tool_file + line.reshape((3,-1)), small_tool_file)
    if rotationMatrixDistance(rotationMatrix) > max_angle:
        max_angle = rotationMatrixDistance(rotationMatrix)
print(max_angle, "degrees")
max_mm = max_angle/5 * 4.2
print(max_mm, "mm")
print("large array")
max_angle = -1
for line in diffs:
    rotationMatrix, translationVector = computeRigidTransform(large_tool_file + line.reshape((3,-1)), large_tool_file)
    if rotationMatrixDistance(rotationMatrix) > max_angle:
        max_angle = rotationMatrixDistance(rotationMatrix)
print(max_angle, "degrees")
max_mm = max_angle/5 * 4.2
print(max_mm, "mm")
print("two array")
max_angle = -1
for line in diffs:
    rotationMatrix, translationVector = computeRigidTransform(double_tool_file + line.reshape((3,-1)), double_tool_file)
    if rotationMatrixDistance(rotationMatrix) > max_angle:
        max_angle = rotationMatrixDistance(rotationMatrix)
print(max_angle, "degrees")
max_mm = max_angle/5 * 4.2
print(max_mm, "mm")
