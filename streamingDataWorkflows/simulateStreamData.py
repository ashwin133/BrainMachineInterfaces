"""
Functionality to simulate the streaming of data from motive by feeding each frame 
"""


# import python specific libraries
import os
import sys
import numpy as np
import pandas as pd
import atexit
import time

print("Program started")
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
        try:
            dataLocation = "../Data/Rishita-jumping jacks 2023-10-18.csv"
            simulatedDF = streamData.extractDataFrameFromCSV(dataLocation = dataLocation, includeCols='Bone Marker')
        except:
            raise Exception('File not found')


# initialise shared memory
shared_Block,sharedArray = streamData.defineSharedMemory(sharedMemoryName= 'Motive Dump', dataType= "Bone Marker", noDataTypes= 25)

print("Starting to dump data into shared memory")
# dump latest data into shared memory
for i in range(0,simulatedDF.shape[0]):
    streamData.dumpFrameDataIntoSharedMemory(simulate=True, simulatedDF= simulatedDF, frame = i, sharedMemArray=sharedArray)
    time.sleep(0.008) # change this later
    print("Dumped Frame {} into shared memory".format(i))
    print(sharedArray)
    
print("Program ended successfully")
shared_Block.close()