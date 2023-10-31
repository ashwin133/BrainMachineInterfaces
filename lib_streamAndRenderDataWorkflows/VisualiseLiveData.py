"""
This file handles workflows needed to visualise data that has been streamed

"""
from multiprocessing import shared_memory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

maxX,maxY,maxZ = -10000,-10000,-10000
minX,minY,minZ = 10000,10000,10000

# add Root Directory to system path to import created packages
sys.path.insert(0,'/Users/rishitabanerjee/Desktop/BrainMachineInterfaces/')
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')

from lib_streamAndRenderDataWorkflows import quaternions

def visualiseFrameData(varsPerDataType,noDataTypes,sharedMemoryName,frameLength = 1000):
    global maxX,maxY,maxZ
    global minX,minY,minZ
    maxX,maxY,maxZ = -10000,-10000,-10000
    minX,minY,minZ = 10000,10000,10000
    if varsPerDataType == 7: # rigid body structure
        i1 = 4
        i2 = 5
        i3 = 6
    else:
        i1 = 2
        i2 = 0
        i3 = 1
    # access the shared memory    
    dataEntries = varsPerDataType * noDataTypes
    SHARED_MEM_NAME = sharedMemoryName
    shared_block = shared_memory.SharedMemory(size= dataEntries * 8, name=SHARED_MEM_NAME, create=False)
    shared_array = np.ndarray(shape=(noDataTypes,varsPerDataType), dtype=np.float64, buffer=shared_block.buf)

    # load the most recent shared memory onto a dataframe
    df = pd.DataFrame(shared_array)

    def update_graph(num):
        # function to update location of points frame by frame
        global maxX,maxY,maxZ
        global minX,minY,minZ
        df = pd.DataFrame(shared_array) 
        print(df)
        graph._offsets3d = (df[i1], df[i2], df[i3])
        title.set_text('Plotting markers, time={}'.format(num))
        maxX, maxY, maxZ = max(df.iloc[:,i1].max(),maxX), max(df.iloc[:,i2].max(),maxY), max(df.iloc[:,i3].max(),maxZ)
        minX, minY, minZ = min(df.iloc[:,i1].min(),minX), min(df.iloc[:,i2].min(),minY), min(df.iloc[:,i3].min(),minZ)

        ax.axes.set_zlim3d(bottom= minZ, top= maxZ) 
        ax.axes.set_xlim3d(left=minX, right=maxX) 
        ax.axes.set_ylim3d(bottom= minY, top=maxY) 

    # set up the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('Plotting markers')

    # plot the first set of data
    graph = ax.scatter(df[i1], df[i2], df[i3])

    # set up the animation
    ani = animation.FuncAnimation(fig, update_graph, frameLength, 
                                interval=8, blit=False)

    plt.show()

def simulateDisplayQuarternionData(varsPerDataType,noDataTypes,sharedMemoryName,frameLength = 1000,sim = False,idxesToPlot = None):
    # varsPerDataType should be 7 for the quaternion data
    # access the shared memory  
    global maxX,maxY,maxZ
    global minX,minY,minZ
    maxX,maxY,maxZ = -10000,-10000,-10000
    minX,minY,minZ = 10000,10000,10000

    if sim == False:
        dataEntries = varsPerDataType * noDataTypes
        SHARED_MEM_NAME = sharedMemoryName
        shared_block = shared_memory.SharedMemory(size= dataEntries * 8, name=SHARED_MEM_NAME, create=False)
        shared_array = np.ndarray(shape=(noDataTypes,varsPerDataType), dtype=np.float64, buffer=shared_block.buf)
        if idxesToPlot is not None:
            df = pd.DataFrame(shared_array[idxesToPlot])
        else:
            df = pd.DataFrame(shared_array)
    else:
        shared_array = np.random.randint(0,5,size = (43,7))
    # load the most recent shared memory onto a dataframe
    
    # we will get structure of database as rigidBody1 - [q_X,q_Y,q_Z,q_W,X,Y,Z]
    if sim == False:
        df_locations = [df.iloc[a,:] for a in range(0,df.shape[0])] # split rows into list
        df_locations_quaternionObjs = [quaternions.quaternionVector(loc = [a.iloc[4]/1000,a.iloc[5]/1000,a.iloc[6]/1000],quaternion=[a.iloc[0],a.iloc[1],a.iloc[2],a.iloc[3]]) for a in df_locations]
        df_directions =  pd.DataFrame([q.qv_mult(q.quaternion,[0,-1,0]) for q in df_locations_quaternionObjs])
        dfPlot = pd.DataFrame({'x':df.iloc[:,4],'y':df.iloc[:,5],'z':df.iloc[:,6],'dirX':df_directions.iloc[:,0],'dirY':df_directions.iloc[:,1],'dirZ':df_directions.iloc[:,2]})
        maxX, maxY, maxZ = max(df.iloc[:,4].max(),maxX), max(df.iloc[:,5].max(),maxY), max(df.iloc[:,6].max(),maxZ)
        minX, minY, minZ = min(df.iloc[:,4].min(),minX), min(df.iloc[:,5].min(),minY), min(df.iloc[:,6].min(),minZ)
    else:
        dfPlot = pd.DataFrame(np.random.randint(0,5,size = (43,4)))
    # for each set of points, find the location of where the vector should point assume for now that it starts in x direction



    def update_graph(num):
        # function to update location of points frame by frame
        global maxX,maxY,maxZ
        global minX,minY,minZ
        if sim == False:
            if idxesToPlot is not None:
                df = pd.DataFrame(shared_array[idxesToPlot])
            else:
                df = pd.DataFrame(shared_array)
            df_locations = [df.iloc[a,:] for a in range(0,df.shape[0])] # split rows into list
            df_locations_quaternionObjs = [quaternions.quaternionVector(loc = [a.iloc[4]/1000,a.iloc[5]/1000,a.iloc[6]/1000],quaternion=[a.iloc[3],a.iloc[0],a.iloc[1],a.iloc[2]]) for a in df_locations]
            df_directions =  pd.DataFrame([q.qv_mult(q.quaternion,[0,-1,0]) for q in df_locations_quaternionObjs])
            dfPlot = pd.DataFrame({'x':df.iloc[:,4],'y':df.iloc[:,5],'z':df.iloc[:,6],'dirX':df_directions.iloc[:,0],'dirY':df_directions.iloc[:,1],'dirZ':df_directions.iloc[:,2]})
            maxX, maxY, maxZ = max(df.iloc[:,6].max(),maxX), max(df.iloc[:,4].max(),maxY), max(df.iloc[:,5].max(),maxZ)
            minX, minY, minZ = min(df.iloc[:,6].min(),minX), min(df.iloc[:,4].min(),minY), min(df.iloc[:,5].min(),minZ)
        else:
            dfPlot = pd.DataFrame(np.random.randint(0,5,size = (43,4)))
        ax.clear()
        print(dfPlot)
        ax.quiver(dfPlot.iloc[:,0], dfPlot.iloc[:,1], dfPlot.iloc[:,2],dfPlot.iloc[:,3],dfPlot.iloc[:,4],dfPlot.iloc[:,5] ,color='r')
        ax.axes.set_zlim3d(bottom= -1.5, top= 2) 
        ax.axes.set_xlim3d(left=-2.5, right=2) 
        ax.axes.set_ylim3d(bottom=-3, top=0) 


    # set up the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    title = ax.set_title('Plotting markers')

    # plot the first set of data
    ax.quiver(dfPlot.iloc[:,0], dfPlot.iloc[:,1], dfPlot.iloc[:,2],dfPlot.iloc[:,3],dfPlot.iloc[:,4],dfPlot.iloc[:,5] ,color='r')
    # set up the animation
    ani = animation.FuncAnimation(fig, update_graph, frameLength, 
                                interval=8, blit=False)

    plt.show()
    



def simulateDisplayQuarternionData_v3(varsPerDataType,noDataTypes,sharedMemoryName,frameLength = 1000,sim = False,idxesToPlot = None):
    # varsPerDataType should be 7 for the quaternion data
    # access the shared memory  
    global maxX,maxY,maxZ
    global minX,minY,minZ
    global quaternionsUnit
    maxX,maxY,maxZ = -10000,-10000,-10000
    minX,minY,minZ = 10000,10000,10000

    if sim == False:
        dataEntries = varsPerDataType * noDataTypes
        SHARED_MEM_NAME = sharedMemoryName
        shared_block = shared_memory.SharedMemory(size= dataEntries * 8, name=SHARED_MEM_NAME, create=False)
        shared_array = np.ndarray(shape=(noDataTypes,varsPerDataType), dtype=np.float64, buffer=shared_block.buf)
        if idxesToPlot is not None:
            df = pd.DataFrame(shared_array[idxesToPlot])
        else:
            df = pd.DataFrame(shared_array)
    else:
        shared_array = np.random.randint(0,5,size = (43,7))
    # load the most recent shared memory onto a dataframe
    
    # we will get structure of database as rigidBody1 - [q_X,q_Y,q_Z,q_W,X,Y,Z]
    if sim == False:
        df_locations = [df.iloc[a,:] for a in range(0,df.shape[0])] # split rows into list
        df_locations_quaternionObjs = [quaternions.quaternionVector(loc = [a.iloc[4]/1000,a.iloc[5]/1000,a.iloc[6]/1000],quaternion=[a.iloc[0],a.iloc[1],a.iloc[2],a.iloc[3]]) for a in df_locations]
        df_directions =  pd.DataFrame([q.qv_mult(q.quaternion,quaternionsUnit[i]) for i,q in enumerate(df_locations_quaternionObjs)])
        dfPlot = pd.DataFrame({'x':df.iloc[:,4],'y':df.iloc[:,5],'z':df.iloc[:,6],'dirX':df_directions.iloc[:,0],'dirY':df_directions.iloc[:,1],'dirZ':df_directions.iloc[:,2]})
        maxX, maxY, maxZ = max(df.iloc[:,4].max(),maxX), max(df.iloc[:,5].max(),maxY), max(df.iloc[:,6].max(),maxZ)
        minX, minY, minZ = min(df.iloc[:,4].min(),minX), min(df.iloc[:,5].min(),minY), min(df.iloc[:,6].min(),minZ)
    else:
        dfPlot = pd.DataFrame(np.random.randint(0,5,size = (43,4)))
    # for each set of points, find the location of where the vector should point assume for now that it starts in x direction



    def update_graph(num):
        # function to update location of points frame by frame
        global maxX,maxY,maxZ
        global minX,minY,minZ
        global quaternionsUnit
        if sim == False:
            if idxesToPlot is not None:
                df = pd.DataFrame(shared_array[idxesToPlot])
            else:
                df = pd.DataFrame(shared_array)
            df_locations = [df.iloc[a,:] for a in range(0,df.shape[0])] # split rows into list
            df_locations_quaternionObjs = [quaternions.quaternionVector(loc = [a.iloc[4]/1000,a.iloc[5]/1000,a.iloc[6]/1000],quaternion=[a.iloc[3],a.iloc[0],a.iloc[1],a.iloc[2]]) for a in df_locations]
            df_directions =  pd.DataFrame([q.qv_mult(q.quaternion,quaternionsUnit[i]) for i,q in enumerate(df_locations_quaternionObjs)])
            dfPlot = pd.DataFrame({'x':df.iloc[:,4],'y':df.iloc[:,5],'z':df.iloc[:,6],'dirX':df_directions.iloc[:,0],'dirY':df_directions.iloc[:,1],'dirZ':df_directions.iloc[:,2]})
            maxX, maxY, maxZ = max(df.iloc[:,6].max(),maxX), max(df.iloc[:,4].max(),maxY), max(df.iloc[:,5].max(),maxZ)
            minX, minY, minZ = min(df.iloc[:,6].min(),minX), min(df.iloc[:,4].min(),minY), min(df.iloc[:,5].min(),minZ)
        else:
            dfPlot = pd.DataFrame(np.random.randint(0,5,size = (43,4)))
        ax.clear()
        print(dfPlot)
        ax.quiver(dfPlot.iloc[:,0], dfPlot.iloc[:,1], dfPlot.iloc[:,2],dfPlot.iloc[:,3],dfPlot.iloc[:,4],dfPlot.iloc[:,5] ,color='r')
        ax.axes.set_zlim3d(bottom= -1.5, top= 2) 
        ax.axes.set_xlim3d(left=-2.5, right=2) 
        ax.axes.set_ylim3d(bottom=-3, top=0) 


    # set up the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    title = ax.set_title('Plotting markers')

    # plot the first set of data
    ax.quiver(dfPlot.iloc[:,0], dfPlot.iloc[:,1], dfPlot.iloc[:,2],dfPlot.iloc[:,3],dfPlot.iloc[:,4],dfPlot.iloc[:,5] ,color='r')
    # set up the animation
    ani = animation.FuncAnimation(fig, update_graph, frameLength, 
                                interval=8, blit=False)

    plt.show()

if __name__ == "__main__":
    

    rigidBodyParts = ['Pelvis', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 
                      'LFArm', 'LHand', 'LThumb1', 'LThumb2', 'LThumb3', 'LIndex1', 'LIndex2', 
                      'LIndex3', 'LMiddle1', 'LMiddle2', 'LMiddle3', 'LRing1', 'LRing2', 'LRing3', 
                      'LPinky1', 'LPinky2', 'LPinky3', 'RShoulder', 'RUArm', 'RFArm', 'RHand', 'RThumb1', 
                      'RThumb2', 'RThumb3', 'RIndex1', 'RIndex2', 'RIndex3', 'RMiddle1', 'RMiddle2', 'RMiddle3', 
                      'RRing1', 'RRing2', 'RRing3', 'RPinky1', 'RPinky2', 'RPinky3', 'LThigh', 'LShin', 
                      'LFoot', 'LToe', 'RThigh', 'RShin', 'RFoot', 'RToe']
    
    rigidBodyPartsDict = {'Pelvis': [1,(0,0,1)], 'Ab': [1,(0,0,1)], 'Chest': [1,(0,-1,0)], 'Neck': [1,(0,0,1)], 'Head': [1,(0,-1,0)], 'LShoulder': [1,(1,0,0)], 'LUArm': [1,(1,0,0)], 
                      'LFArm': [1,(1,0,0)], 'LHand': [1,(1,0,0)], 'LThumb1': [0], 'LThumb2': [0], 'LThumb3': [0], 'LIndex1': [0], 'LIndex2': [0], 
                      'LIndex3': [0], 'LMiddle1': [0], 'LMiddle2': [0], 'LMiddle3': [0], 'LRing1': [0], 'LRing2': [0], 'LRing3': [0], 
                      'LPinky1': [0], 'LPinky2': [0], 'LPinky3': [0], 'RShoulder': [1,(-1,0,0)], 'RUArm': [1,(-1,0,0)], 'RFArm': [1,(-1,0,0)], 'RHand': [1,(-1,0,0)], 'RThumb1': [0], 
                      'RThumb2': [0], 'RThumb3': [0], 'RIndex1': [0], 'RIndex2': [0], 'RIndex3': [0], 'RMiddle1': [0], 'RMiddle2': [0], 'RMiddle3': [0], 
                      'RRing1': [0], 'RRing2': [0], 'RRing3': [0], 'RPinky1': [0], 'RPinky2': [0], 'RPinky3': [0], 'LThigh': [0], 'LShin': [0], 
                      'LFoot': [0], 'LToe': [0], 'RThigh': [0], 'RShin': [0], 'RFoot': [0], 'RToe': [0]}
    
    simpleBodyParts = [0,1,2,3,4,5,6,7,8,24,25,26,27]

    renderingBodyParts = []
    renderingBodyPartsIdxes = []
    counter = 0
    for bodyPart in rigidBodyPartsDict:
        if rigidBodyPartsDict[bodyPart][0] == 1:
            renderingBodyParts.append(bodyPart)
            renderingBodyPartsIdxes.append(counter)
        counter += 1
    quaternionsUnit = [rigidBodyPartsDict[bodyPart][1] for bodyPart in renderingBodyParts]

    #visualiseFrameData(varsPerDataType=7,noDataTypes=51,sharedMemoryName= 'Test Rigid Body')
    #simulateDisplayQuarternionData(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = simpleBodyParts) # used 25 width for offline
    simulateDisplayQuarternionData(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [0]) # pelvis [0,0,1]
    #simulateDisplayQuarternionData(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [1]) # ab [0,0,1]
    #simulateDisplayQuarternionData(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [2]) # chest [0,-1,0]
    #simulateDisplayQuarternionData(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [3]) # neck [0,0,1]
    #simulateDisplayQuarternionData(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [4]) # Head [0,-1,0]
    #simulateDisplayQuarternionData(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [5]) # left shoulder quaternion of [1,0,0]
    #simulateDisplayQuarternionData(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [6]) # left upper arm, quaternion of [1,0,0]
    #simulateDisplayQuarternionData(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [7]) # left forearm, quaternion of [1,0,0]
    #simulateDisplayQuarternionData(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [8]) # left hand quaternion of [1,0,0]
    #simulateDisplayQuarternionData(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [24]) # right shoulder quaternion [-1,0,0]
    #simulateDisplayQuarternionData(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [25]) # right upper arm, quaternion of [-1,0,0]
    #simulateDisplayQuarternionData(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [26]) # right forearm, quaternion of [-1,0,0]
    #simulateDisplayQuarternionData(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [27]) # right hand, quaternion of [-1,0,0]
    
    simulateDisplayQuarternionData_v3(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = simpleBodyParts)
    print("Program ended")
