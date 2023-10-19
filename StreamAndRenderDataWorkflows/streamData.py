"""
This file contains the functionality to stream live data from the motive computer and store in shared memory
There is also functionality to simulate streaming data

"""

import os
import sys
import pandas as pd

 

def fetchLiveData(simulate = False):
    if simulate:
        # this will simulate the process of retrieving live data by retrieving the frame corresponding to the current timestamp 
        pass 
    pass


def defineSharedMemory(sharedMemoryName = 'MotiveDump'):
    pass

def dumpFrameDataIntoSharedMemory():
    pass

def retrieveSharedMemoryData(sharedMemoryName = 'MotiveDump'):
    pass

def extractDataFrameFromCSV(dataLocation):

  # extract the experimental data onto a df, test file will check whether 
  # rows skipped will need to be updated in the future
  df = pd.read_csv(dataLocation,skiprows=[0,1,2,4],header = None)

  # the top row has the names of each part so extract this
  bodyParts = df.iloc[0].values
  # extract the kinematic nature of each column (rotation or position)
  kinematicType = df.iloc[1].values
  # extract the variable in third row
  kinematicVariable = df.iloc[2].values

  # create a header array to store a simplified header for each column
  headerArray = []
  headerArray.append('Frame')
  headerArray.append('Time (Seconds)')


  for i in range(2,df.shape[1]):
    headerArray.append(bodyParts[i] + ' ' + kinematicType[i] + ' ' + kinematicVariable[i])


  # now create dataframe removing the previous rows of metadata and reassigning the
  # column titles

  df = df.iloc[3:]
  df.columns = headerArray
  print(df.head())
  df = df.astype(float)
  return df