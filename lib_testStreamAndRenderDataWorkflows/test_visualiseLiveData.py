"""
Contains tests for workflows related to visualising live data
"""
# import standard python libraries
import sys
import pytest
import warnings
import pytest
import os

# add Root Directory to system path to import created packages
try:
    sys.path.insert(0,'/Users/rishitabanerjee/Desktop/BrainMachineInterfaces/')
except ModuleNotFoundError:
    try: 
        sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')
    except:
        pass

# import user based libraries
from lib_streamAndRenderDataWorkflows import VisualiseLiveData,streamData


def testVisualiseLiveData():
    pass
    # implement later
    