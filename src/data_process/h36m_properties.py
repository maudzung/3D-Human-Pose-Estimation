import numpy as np
from glob import glob
import os
import cv2

# Define the list of 17 joints
h36m_keypoints = {
    0: 'Hip',
    1: 'RHip',
    2: 'RKnee',
    3: 'RFoot',
    6: 'LHip',
    7: 'LKnee',
    8: 'LFoot',
    12: 'Spine',
    13: 'Neck',
    14: 'Nose',
    15: 'Head',
    17: 'LShoulder',
    18: 'LElbow',
    19: 'LWrist',
    25: 'RShoulder',
    26: 'RElbow',
    27: 'RWrist',
}
h36m_parent_ids = np.array([0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15], dtype=np.int)

# connections (bones) of our representation
h36m_connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                    [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                    [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
# Left / right indicator (except Hip)
h36m_lr = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

if __name__ == '__main__':
    for child_id_idx, child_id in enumerate(h36m_keypoints.keys()):
        parent_id = h36m_parent_ids[child_id_idx]
        child_name = h36m_keypoints[child_id]
        parent_name = list(h36m_keypoints.values())[parent_id]

        print('parent: {} - child: {}'.format(parent_name, child_name))