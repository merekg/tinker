import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os

# constants
PATIENT_REFERENCE_MARKERS = np.array([[-40.00,29.50,-11.64],
                                      [30.00,47.00,-11.64],
                                      [20.00,-20.50,-11.64],
                                      [-35.00,-10.50,-11.64]])

PROBE_MARKERS = np.array([[-159.44,-1.16,-0.71],
                          [-229.29,-1.16,-0.71],
                          [-255.45,-43.83,-0.71],
                          [-263.07,42.27,-0.71],
                          [-304.47,-1.16,-0.71]])

SYSTEM_MARKERS = np.array([[0.00,0.00,0.00],
                           [58.18,-34.13,0.00],
                           [145.75,0.00,0.00],
                           [197.23,58.27,144.66],
                           [160.07,255.41,147.57],
                           [215.51,197.59,145.73]])

NORMAL_SYSTEM_ORIENTATION_QUAT = np.array([0.950122185,-0.0488879,0.27195438,-0.144632525])
IMAGE_DETECTOR_POSE = np.array([0.56978,-0.713141,-0.408389,-163.879,0.628265,0.698345,-0.342924,-324.396,0.529749,-0.0611854,0.845944,685.658,0,0,0,1]).reshape((4,4))
M = -4/11
B = 130/11

DETECTOR_LAST_VISIBLE = np.array([-28.5736,-155.322,130.825])

def invert_matrix(T):
    r = T[:3, :3]
    t = T[:3, 3]
    ri = np.linalg.inv(r)
    ti = - np.matmul(ri, t)
    Ti = np.zeros_like(T)
    Ti[:3,:3] = ri
    Ti[:3,3] = ti
    Ti[3,3] = 1
    return Ti

def get_marker_poses(q,t,tool_file):
    tool_rotation = R.from_quat(np.array([q[1],q[2],q[3],q[0]])).as_matrix()
    marker_poses = np.array([tool_rotation @ marker + t for marker in markers])
    return marker_poses

def apply_tx(target, tx):
    return (tx @ np.append(target,[1]))[:3]

def main():

if __name__ == "__main__":
    main()
