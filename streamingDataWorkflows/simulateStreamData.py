"""
Functionality to simulate the streaming of data from motive by feeding each frame 
"""


# import python specific libraries
import os
import sys
import numpy as np
import pandas as pd

# add base dir to system path
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')


# import my libraries
from lib_streamAndRenderDataWorkflows import streamData, VisualiseLiveData

# set what type of data to get e.g. Bone, Bone Marker
typeData = "Bone Marker"

# feed in location of csv data to extract dataframe 
try:        
    dataLocation = "Data/Rishita-jumping jacks 2023-10-18.csv"
    simulatedDF = streamData.extractDataFrameFromCSV(dataLocation = dataLocation,includeCols='Bone Marker')
except FileNotFoundError: # if file is run from location of file this is needed
    try:
        dataLocation = "Rishita-jumping jacks 2023-10-18.csv"
        simulatedDF = streamData.extractDataFrameFromCSV(dataLocation = dataLocation, includeCols='Bone Marker')
    except:
        raise Exception('File not found')
    
print("Program ended successfully")