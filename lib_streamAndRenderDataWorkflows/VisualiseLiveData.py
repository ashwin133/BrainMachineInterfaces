"""
This file handles workflows needed to visualise data that has been streamed

"""
from multiprocessing import shared_memory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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