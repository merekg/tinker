import numpy as np

DELTA = 10

def get_point_signature(point, cloud):
    distances = []
    for p in cloud:
        if not all(point==p):
            distances.append(np.linalg.norm(point-p))
    return np.array(sorted(distances))

array = np.array([[0,90,0],[250,94.4,0],[130,115,0],[12,189,0],[140,200,0],[220,183.5,0]])
signatures = []
for point in array:
    signatures.append(get_point_signature(point,array))
signatures = np.array(signatures)
s_min = signatures - DELTA
s_max = signatures + DELTA
print(s_min)
print(s_max)

for i1 in range(len(signatures)):
    for i2 in range(len(signatures)):
        t1 = s_min[i1] + 0.1
        t2 = s_min[i2] + 0.1
        if (all(t1 < s_max[i2]) and all(t1 > s_min[i2])) or (all(t2 < s_max[i1]) and all(t2 > s_max[i1])):
            print("NOPE")
            print(i1, i2)
