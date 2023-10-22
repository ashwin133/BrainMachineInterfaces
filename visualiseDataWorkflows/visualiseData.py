"""
Enabling functionality to render data 
"""
import numpy as np
import matplotlib.pyplot as plt    
import matplotlib.animation as animation
from multiprocessing import shared_memory
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def visualise2DDataFrom3DarrayAnimation(sharedMemoryName = None,noDataTypes = None, varsPerDataType = None):

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
        graph._offsets3d = (df[0], df[1], df[2])
        title.set_text('Plotting markers, time={}'.format(num))

    # set up the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('Plotting markers')

    # plot the first set of data
    graph = ax.scatter(df[0], df[1], df[2])

    # set up the animation
    ani = animation.FuncAnimation(fig, update_graph, 1000, 
                                interval=8, blit=False)

    plt.show()


if __name__ == "__main__":
    visualise2DDataFrom3DarrayAnimation(sharedMemoryName= 'Motive Dump',noDataTypes=25,varsPerDataType=3)