"""
script to test rotation algorithm to implement for 
calibrating game

"""

# import libraries
import numpy as np 
import matplotlib.pyplot as plt
from lib_streamAndRenderDataWorkflows.config_streaming import *

from multiprocessing import shared_memory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import time
from time import perf_counter_ns


def testCalibration_1():
    """
    first proof of concept for calibrating off axis vectors to on axis vectors
    only shows how matrix transform occurs
    """
    # define a vector for now
    unitAxisX = np.zeros(3)
    unitAxisX[0] = 1

    # define the new frame of reference x vector to push to normalised x
    offAxisX = np.zeros(3)
    offAxisX[0] = 0.7
    offAxisX[1] = -0.5

    # calculate thetha from dot product
    thetha_rad = np.arccos(np.dot(unitAxisX,offAxisX)/(np.linalg.norm(unitAxisX) * np.linalg.norm(offAxisX)))

    thetha_deg = np.rad2deg(thetha_rad)
    print(thetha_deg)

    # calculate Q
    Q = np.zeros((3,3))
    Q[0,0] = np.cos(thetha_rad)
    Q[1,1] = Q[0,0]
    Q[0,1] = np.sin(thetha_rad)
    Q[1,0] = - Q[0,1]
    Q[2,2] = 1

    unitAxisY = np.zeros(3)
    unitAxisY[1] = 1

    offAxisY = np.zeros(3)
    offAxisY[0] = 0.5
    offAxisY[1] = 0.7


    y_new = np.matmul(Q.transpose(),offAxisY)
    x_new = np.matmul(Q.transpose(),offAxisX)
    #plotting
    point1 = np.zeros(3)
    point1[0] = 0.5
    point1[1] = 0.7
    dir1 = np.zeros(3)
    dir1[0] = 0.7
    dir1[1] = -0.5
    point1_new = np.matmul(Q.transpose(),point1)
    dir1_new = np.matmul(Q.transpose(),dir1)

    # point 2
    point2 = np.zeros(3)
    point2[0] = 0.5
    point2[1] = -0.7
    point2[2] = 0.5
    dir2 = np.zeros(3)
    dir2[0] = -0.7
    dir2[1] = -0.5
    dir2[2] = 0.5
    point2_new = np.matmul(Q.transpose(),point2)
    dir2_new = np.matmul(Q.transpose(),dir2)
    # z_1 = np.array([0.2,0.9,0])
    # z_2 = np.matmul(Q,z_1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(0,0,0,*offAxisX,color='b')
    ax.quiver(0,0,0,*offAxisY,color='b')
    ax.quiver(0,0,0,*x_new,color='g')
    ax.quiver(0,0,0,*y_new,color='g')
    ax.quiver(0,0,0,*unitAxisX,color='r')
    ax.quiver(0,0,0,*unitAxisY,color='r')
    ax.quiver(*point2,*dir2,color = 'y')
    ax.quiver(*point2_new,*dir2_new,color = 'm')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    plt.show()

#testCalibration_1()
def calcCalibrationConstants(calibrationToVector, calibrationFromVector):
    """
    attempts to calibrate for person standing off x axis by finding the transformation
    matrix to transform off axis motion to the standard axes
    returns a transformation matrix that can convert directions and positions 
    """
    # calculate thetha from dot product
    thetha_rad = np.arccos(np.dot(calibrationToVector,calibrationFromVector)/(np.linalg.norm(calibrationToVector) * np.linalg.norm(calibrationFromVector)))

    # thetha_deg = np.rad2deg(thetha_rad)
    # print(thetha_deg)

    # calculate Q
    Q = np.zeros((3,3))
    Q[0,0] = np.cos(thetha_rad)
    Q[1,1] = Q[0,0]
    Q[0,1] = np.sin(thetha_rad)
    Q[1,0] = - Q[0,1]
    Q[2,2] = 1

    return Q.transpose()

def liveCalibrationTest(varsPerDataType,noDataTypes,sharedMemoryName,calibrateFromRightHand = True):
    print('Prepare for calibration ...')
    time.sleep(4)
    print('Calibrating ...')
    SHARED_MEM_NAME = sharedMemoryName
    dataEntries = varsPerDataType * noDataTypes
    shared_block = shared_memory.SharedMemory(size= dataEntries * 8, name=SHARED_MEM_NAME, create=False)
    shared_array = np.ndarray(shape=(noDataTypes,varsPerDataType), dtype=np.float64, buffer=shared_block.buf)
    if calibrateFromRightHand:
        df = np.array(shared_array[27,:]) # idx of right hand
    else:
        raise('Need to code this in')
    calibrationFromVector = df[3:6]
    calibrationToVector = np.array([1,0,0])
    Q_t = calcCalibrationConstants(calibrationToVector,calibrationFromVector)
    return Q_t


maxX,maxY,maxZ = 0,0,0
minX,minY,minZ = 0,0,0
calMatrix = None
def renderBodyVectors(varsPerDataType,noDataTypes,sharedMemoryName,frameLength = 1000,sim = False,idxesToPlot = None):
    # varsPerDataType should be 7 for the quaternion data
    # access the shared memory  
    """
    use this ver for testing if data is processed at time of data streaming
    """
    global maxX,maxY,maxZ
    global minX,minY,minZ
    global quaternionsUnit
    global colourCode

    maxX,maxY,maxZ = -10000,-10000,-10000
    minX,minY,minZ = 10000,10000,10000

    if sim == False:
        dataEntries = varsPerDataType * noDataTypes
        SHARED_MEM_NAME = sharedMemoryName
        shared_block = shared_memory.SharedMemory(size= dataEntries * 8, name=SHARED_MEM_NAME, create=False)
        shared_array = np.ndarray(shape=(noDataTypes,varsPerDataType), dtype=np.float64, buffer=shared_block.buf)
        df = pd.DataFrame(shared_array[idxesToPlot])
        for i in range(df.shape[0]):
            df.iloc[i,0:3] = np.matmul(Q_t,df.iloc[i,0:3])
            df.iloc[i,3:6] = np.matmul(Q_t,df.iloc[i,3:6]) 
        maxX, maxY, maxZ = max(df.iloc[:,4].max(),maxX), max(df.iloc[:,5].max(),maxY), max(df.iloc[:,6].max(),maxZ)
        minX, minY, minZ = min(df.iloc[:,4].min(),minX), min(df.iloc[:,5].min(),minY), min(df.iloc[:,6].min(),minZ)
    # for each set of points, find the location of where the vector should point assume for now that it starts in x direction



    def update_graph(num):
        # function to update location of points frame by frame
        global maxX,maxY,maxZ
        global minX,minY,minZ
        global quaternionsUnit
        global colourCode
        global Q_t
        if sim == False:
            df = pd.DataFrame(shared_array[idxesToPlot])
            #calibration
            for i in range(df.shape[0]):
                df.iloc[i,0:3] = np.matmul(Q_t,df.iloc[i,0:3])
                df.iloc[i,3:6] = np.matmul(Q_t,df.iloc[i,3:6])
            maxX, maxY, maxZ = max(df.iloc[:,0].max(),maxX), max(df.iloc[:,1].max(),maxY), max(df.iloc[:,2].max(),maxZ)
            minX, minY, minZ = min(df.iloc[:,0].min(),minX), min(df.iloc[:,1].min(),minY), min(df.iloc[:,2].min(),minZ)

        ax.clear()
        if num % 20:
            print(pd.DataFrame(shared_array[27]))
            print('thetha_z',np.arctan(df.iloc[-1,4]/df.iloc[-1,3]))
            print('thetha_y',np.arctan(df.iloc[-1,3]/df.iloc[-1,5]))
            print('thetha_x',np.arctan(df.iloc[-1,4]/df.iloc[-1,5]))
            #print(minX,maxX,minY,maxY,minZ,maxZ)

        ax.quiver(df.iloc[-1,0], df.iloc[-1,1], df.iloc[-1,2],df.iloc[-1,3],df.iloc[-1,4],df.iloc[-1,5] ,color=colourCode[-1])
        ax.axes.set_zlim3d(bottom= minZ*1.2, top= maxZ*1.2) 
        ax.axes.set_xlim3d(left=minX*1.2, right=maxY*1.2) 
        ax.axes.set_ylim3d(bottom=minY*1.2, top=maxY*1.2) 


    # set up the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    title = ax.set_title('3D MOTION VISUALISATION')
    ax.quiver(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2],df.iloc[:,3],df.iloc[:,4],df.iloc[:,5] ,color=colourCode)
    ani = animation.FuncAnimation(fig, update_graph, frameLength, 
                                interval=8, blit=False)
    plt.show()







varsPerDataType = 7
noDataTypes = 51
sharedMemoryName = 'Test Rigid Body'
SHARED_MEM_NAME = sharedMemoryName

Q_t = liveCalibrationTest(varsPerDataType,noDataTypes,sharedMemoryName,calibrateFromRightHand=True)
renderBodyVectors(varsPerDataType,noDataTypes,sharedMemoryName,idxesToPlot = simpleBodyParts) # use v5 for post processed data