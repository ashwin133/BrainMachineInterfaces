"""
This file contains the functionality to stream live data from the motive computer and store in shared memory
There is also functionality to simulate streaming data

"""
# import python specific libraries
import os
import sys
import pandas as pd
import numpy as np
from multiprocessing import shared_memory
import atexit
import time
 

def fetchLiveData(sharedMemoryLocation,simulate = False,simulatedDF = None):
    """
    This function is designed to run continuously in the background and simulates the client which fetches
    data from motive and dumps it in shared memory.
    """

    if simulatedDF is None:
        raise Exception("Simulated Dataframe data not provided but the fetch live data simulator is called")

    if simulate:
        # this will simulate the process of retrieving live data by retrieving the frame corresponding to the current timestamp 
        
        x_headers = []
        y_headers = []
        z_headers = []
        for header in simulatedDF.columns:
            if 'X' in header:
                x_headers.append(header)
            if 'Y' in header:
                y_headers.append(header)
            if 'Z' in header:
                z_headers.append(header)

        headers = [x_headers, y_headers, z_headers]
        t_start = time.time()
        current_frame = 0
        n = 0.1
        c = 0

        while True:

            # find current timestamp from start of simulation
            timestamp = '%.6f'%(time.time() - t_start)

            # if data is over, keep returning the last frame
            if timestamp > simulatedDF['Time'].max():
                current_frame_data = simulatedDF[current_frame]
            
            # retrieve data for current timestamp
            if timestamp in simulatedDF.loc[:, 'Time(seconds)']:
                current_frame = np.where(simulatedDF['Time(seconds)'] == timestamp)
                current_frame_data = simulatedDF[current_frame]

            # every nth of a second push a frame to shared memory, currently assuming Bone Marker data type
            if timestamp == n*c:
                idx = 0

                for header in headers:
                    sharedMemoryLocation[idx, :] = current_frame_data[header].to_numpy()
                    idx += 1
                c+=1

    else: # functionality for fetching actual data off motive
        pass 



def defineSharedMemory(sharedMemoryName = 'MotiveDump',dataType = 'Bone Marker',noDataTypes = 25):
    """
    Initialise shared memory

    @PARAM: sharedMemoryName - name to initialise the shared memory
    @PARAM: dataType - type of marker being looked at - e.g. Bone, Bone Marker
    @PARAM: noDataTypes - number of each type of marker, e.g. if bone marker selected then in an
    upper skeleton there are 25
    """
    varsPerDataType = None
    if dataType == "Bone Marker":
        varsPerDataType = 3 # doesn't have rotations, only x,y,z
    elif dataType == "Bone":
        varsPerDataType = 7 # 4 rotations and 3 positions
    dataEntries = varsPerDataType * noDataTypes # calculate how many data entries needed for each timestamp

    SHARED_MEM_NAME = sharedMemoryName
    shared_block = shared_memory.SharedMemory(size= dataEntries * 8, name=SHARED_MEM_NAME, create=True)
    shared_array = np.ndarray(shape=(noDataTypes,varsPerDataType), dtype=np.float64, buffer=shared_block.buf)
    return shared_block,shared_array

def dumpFrameDataIntoSharedMemory(simulate = False,simulatedDF = None,frame = 0,sharedMemArray = None):
    if simulate:
        rowData = simulatedDF.iloc[frame,:][2:]
        lengthRowData = rowData.shape[0]
        noTypes,noDims = sharedMemArray.shape
        count = 0
        i = 0
        while count < lengthRowData:
            for j in range(0,noDims):
                sharedMemArray[i][j] = rowData[count+j]
            i += 1
            count += noDims





def retrieveSharedMemoryData(sharedMemoryName = 'MotiveDump'):
    pass

def extractDataFrameFromCSV(dataLocation,includeCols = 'Bone Marker'):
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