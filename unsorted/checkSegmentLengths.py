import sys
import numpy as np

marker_poses = np.loadtxt(sys.argv[1], delimiter=",").reshape((-1,3))
marker_poses2 = marker_poses

# minimum distance handled here
distances = np.array([-.5])

# gather all segment lengths
for m1 in marker_poses:
    for m2 in marker_poses2:
        if not np.array_equal(m1,m2):
            distances = np.append(distances, np.linalg.norm(m1 - m2))

# remove duplicate distances, check that we have the correct number of lengths
distances = np.unique(distances)
expected_distance_length = 15
if len(distances[1:]) is not expected_distance_length:
    print("wrong number of distances.", len(distance[1:]))
    exit()

# find the differences in the distances
differences = np.array([-1])
distances2 = distances
for d1 in distances:
    for d2 in distances:
        if not d1 == d2:
            differences = np.append(differences, np.absolute(d1 - d2))

# remove duplicate differences
expected_length = 15 +14 + 13+ 12+ 11+ 10 + 9 + 8 +7 +6 +5 +4 +3 +2 +1
differences = np.unique(differences)
if expected_length is not len(differences[1:]):
    print("PROBLEM")
    print( expected_length)
    print(len(differences[1:]))
    exit()

print(np.sort(differences[1:]))
