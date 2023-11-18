"""
file to check if it can read test data from experiment
"""

import numpy as np

data = np.load('liveData6.npz')

print(data['targetBoxAppearTimes'])
print(data['targetBoxHitTimes'])
