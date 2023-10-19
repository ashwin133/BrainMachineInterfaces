"""
This file contains the functionality to stream live data from the motive computer and store in shared memory
There is also functionality to simulate streaming data

"""

import os
import sys
import pandas as pd

 

def fetchLiveData(sharedMemoryLocation,simulate = False,simulatedDF = None):
    """
    This function is designed to run continuously in the background and simulates the client which fetches
    data from motive and dumps it in shared memory.
    """
    if simulate:
        # this will simulate the process of retrieving live data by retrieving the frame corresponding to the current timestamp 
        if simulatedDF is None:
            raise Exception("Simulated Dataframe data not provided but the fetch live data simulator is called")

            # every nth of a second push a frame to shared memory

    else: # functionality for fetching actual data off motive
        pass 

    pass


def defineSharedMemory(sharedMemoryName = 'MotiveDump'):
    pass

def dumpFrameDataIntoSharedMemory():
    pass

def retrieveSharedMemoryData(sharedMemoryName = 'MotiveDump'):
    pass

def extractDataFrameFromCSV(dataLocation,includeCols = None):
    """
    @PARAM: dataLocation: relative path to csv data
    @PARAM: includeCols: Includes columns of a specific type, e.g. Bone, Bone Marker

    RETURN: a dataframe 
    """

    # extract the experimental data onto a df, test file will check whether 
    # rows skipped will need to be updated in the future
    df = pd.read_csv(dataLocation,skiprows=[0,1,4],header = None)

    # the first row contains the type of each marker, i.e. marker/bone etc.
    markerType = df.iloc[0].values
    # the second row has the names of each part so extract this
    bodyParts = df.iloc[1].values
    # extract the kinematic nature of each column (rotation or position)
    kinematicType = df.iloc[2].values
    # extract the variable in fourth row
    kinematicVariable = df.iloc[3].values

    # create a header array to store a simplified header for each column
    headerArray = []
    headerArray.append('Frame')
    headerArray.append('Time (Seconds)')
    
    # create an index to find when to truncate column
    colStartTruncateIndex = None
    colEndTruncateIndex = None


    for i in range(2,df.shape[1]):
        currHeader = bodyParts[i] + ' ' + kinematicType[i] + ' ' + kinematicVariable[i]
        headerArray.append(currHeader)
        if includeCols == None or includeCols in markerType[i]:
            if colStartTruncateIndex == None:
                colStartTruncateIndex = i
        elif colStartTruncateIndex is not None and colEndTruncateIndex == None:
            colEndTruncateIndex = i


    # now create dataframe removing the previous rows of metadata and reassigning the
    # column titles

    # include only the frame data
    df = df.iloc[4:]

    # rename columns to a more descriptive label: body part, kinematic type, kinematic variable
    df.columns = headerArray
    df = df.astype(float)
    if includeCols != None:
        df_firstCols = df.iloc[:,:2]
        if colStartTruncateIndex is None: colStartTruncateIndex, colEndTruncateIndex = 0,0
        df = df.iloc[:,colStartTruncateIndex:colEndTruncateIndex]
        df = pd.concat([df_firstCols,df],axis=1)

    return df