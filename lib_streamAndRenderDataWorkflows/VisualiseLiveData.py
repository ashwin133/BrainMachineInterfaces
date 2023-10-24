"""
This file handles workflows needed to visualise data that has been streamed

"""
from multiprocessing import shared_memory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# add Root Directory to system path to import created packages
sys.path.insert(0,'/Users/rishitabanerjee/Desktop/BrainMachineInterfaces/')
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')

from lib_streamAndRenderDataWorkflows import quaternions

def visualiseFrameData(varsPerDataType,noDataTypes,sharedMemoryName,frameLength = 1000):
    # access the shared memory    
    dataEntries = varsPerDataType * noDataTypes
    SHARED_MEM_NAME = sharedMemoryName
    shared_block = shared_memory.SharedMemory(size= dataEntries * 8, name=SHARED_MEM_NAME, create=False)
    shared_array = np.ndarray(shape=(noDataTypes,varsPerDataType), dtype=np.float64, buffer=shared_block.buf)

    # load the most recent shared memory onto a dataframe
    df = pd.DataFrame(shared_array)

    def update_graph(num):
        # function to update location of points frame by frame
        df = pd.DataFrame(shared_array) 
        print(df)
        graph._offsets3d = (df[2], df[0], df[1])
        title.set_text('Plotting markers, time={}'.format(num))

    # set up the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('Plotting markers')

    # plot the first set of data
    graph = ax.scatter(df[2], df[0], df[1])

    # set up the animation
    ani = animation.FuncAnimation(fig, update_graph, frameLength, 
                                interval=8, blit=False)

    plt.show()

def simulateDisplayQuarternionData(varsPerDataType,noDataTypes,sharedMemoryName,frameLength = 1000,sim = False):
    # varsPerDataType should be 7 for the quaternion data
    # access the shared memory    

    if sim == False:
        dataEntries = varsPerDataType * noDataTypes
        SHARED_MEM_NAME = sharedMemoryName
        shared_block = shared_memory.SharedMemory(size= dataEntries * 8, name=SHARED_MEM_NAME, create=False)
        shared_array = np.ndarray(shape=(noDataTypes,varsPerDataType), dtype=np.float64, buffer=shared_block.buf)
        df = pd.DataFrame(shared_array)
    else:
        shared_array = np.random.randint(0,5,size = (43,7))
    # load the most recent shared memory onto a dataframe
    
    # we will get structure of database as rigidBody1 - [q_X,q_Y,q_Z,q_W,X,Y,Z]
    if sim == False:
        df_locations = [df.iloc[a,:] for a in range(0,df.shape[0])] # split rows into list
        df_locations_quaternionObjs = [quaternions.quaternionVector(loc = [a.iloc[4]/1000,a.iloc[5]/1000,a.iloc[6]/1000],quaternion=[a.iloc[3],a.iloc[0],a.iloc[1],a.iloc[2]]) for a in df_locations]
        df_directions =  pd.DataFrame([q.qv_mult(q.quaternion,[1,0,0]) for q in df_locations_quaternionObjs])
        dfPlot = pd.DataFrame({'x':df.iloc[:,4],'y':df.iloc[:,5],'z':df.iloc[:,6],'dirX':df_directions.iloc[:,0]+df.iloc[:,4],'dirY':df_directions.iloc[:,1]+df.iloc[:,5],'dirZ':df_directions.iloc[:,2]+df.iloc[:,6]})
    else:
        dfPlot = pd.DataFrame(np.random.randint(0,5,size = (43,4)))
    # for each set of points, find the location of where the vector should point assume for now that it starts in x direction

    def update_graph(num):
        # function to update location of points frame by frame
        
        if sim == False:
            df = pd.DataFrame(shared_array)
            df_locations = [df.iloc[a,:] for a in range(0,df.shape[0])] # split rows into list
            df_locations_quaternionObjs = [quaternions.quaternionVector(loc = [a.iloc[4]/1000,a.iloc[5]/1000,a.iloc[6]/1000],quaternion=[a.iloc[3],a.iloc[0],a.iloc[1],a.iloc[2]]) for a in df_locations]
            df_directions =  pd.DataFrame([q.qv_mult(q.quaternion,[1,0,0]) for q in df_locations_quaternionObjs])
            dfPlot = pd.DataFrame({'x':df.iloc[:,4],'y':df.iloc[:,5],'z':df.iloc[:,6],'dirX':df_directions.iloc[:,0]+df.iloc[:,4],'dirY':df_directions.iloc[:,1]+df.iloc[:,5],'dirZ':df_directions.iloc[:,2]+df.iloc[:,6]})
        else:
            dfPlot = pd.DataFrame(np.random.randint(0,5,size = (43,4)))
        ax.clear()
        ax.quiver(dfPlot.iloc[:,0], dfPlot.iloc[:,1], dfPlot.iloc[:,2],dfPlot.iloc[:,3],dfPlot.iloc[:,4],dfPlot.iloc[:,5] ,color='r')
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
    simulateDisplayQuarternionData(7,43,sharedMemoryName= 'Test Rigid Body')