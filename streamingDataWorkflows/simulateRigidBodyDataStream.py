"""
Functionality to simulate the streaming of rigid body data from motive by feeding each frame 
"""


# import python specific libraries
import os
import sys
import numpy as np
import pandas as pd
import atexit
import time

sys.path.insert(0,'/Users/rishitabanerjee/Desktop/BrainMachineInterfaces/')
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')


# Import functionalities from other files
import lib_streamAndRenderDataWorkflows.Client.NatNetClient as NatNetClient
import lib_streamAndRenderDataWorkflows.Client.DataDescriptions as DataDescriptions
import lib_streamAndRenderDataWorkflows.Client.MoCapData as MoCapData
import lib_streamAndRenderDataWorkflows.Client.PythonSample as PythonSample

print("Program started")
# add base dir to system path


# import my libraries
from lib_streamAndRenderDataWorkflows import streamData, VisualiseLiveData

# set what type of data to get e.g. Bone, Bone Marker
typeData = "Bone Marker"

# Try to access file         
try:        
    dataLocation = "Data/presentation_demo_rigidBodies.csv"
    simulatedDF = streamData.extractDataFrameFromCSV(dataLocation = dataLocation,includeCols='Bone',preprocessed=False)
except FileNotFoundError: # if file is run from location of file this is needed
    try:
        dataLocation = "presentation_demo_rigidBodies.csv"
        simulatedDF = streamData.extractDataFrameFromCSV(dataLocation = dataLocation, includeCols='Bone',preprocessed=False)
    except:
        try:
            dataLocation = "../Data/presentation_demo_rigidBodies.csv"
            simulatedDF = streamData.extractDataFrameFromCSV(dataLocation = dataLocation, includeCols='Bone',preprocessed=False)
        except:
            raise Exception('File not found')


# Location of game save:
gameSaveLocation = "../Experiment_pointer/PointerExperimentData/Ashwin_12_01__19_57_trial1_training1.npz"
data = np.load(gameSaveLocation,allow_pickle=True)
simulatedDF= data['allBodyPartsData']
simulatedDF = pd.DataFrame(simulatedDF)
# initialise shared memory
shared_Block,sharedArray = streamData.defineSharedMemory(sharedMemoryName= 'Test Rigid Body', dataType= "Bone", noDataTypes= 51)


print("Starting to dump data into shared memory")
#dump latest data into shared memory
for i in range(0,simulatedDF.shape[0]):
    streamData.dumpFrameDataIntoSharedMemory(simulate=True, simulatedDF= simulatedDF, frame = i, sharedMemArray=sharedArray,preprocessedSharedArray=True)
    time.sleep(0.0125) # change this later
    if i%100 == 0:
        print("Dumped Frame {} into shared memory".format(i))
        print(sharedArray)

#streamData.fetchLiveData(shared_Array, shared_Block, simulate=False, simulatedDF=simulated_DF)
    
print("Program ended successfully")
shared_Block.close()