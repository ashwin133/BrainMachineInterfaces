"""
Contains tests for workflows involved in streaming data from motive and related to shared memory
"""
# import standard python libraries
import sys

# add Root Directory to system path to import created packages
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')

# import created packages
from StreamAndRenderDataWorkflows.streamData import extractDataFrameFromCSV


def testExtractDataFrameFromCSV():
    dataLocation = "../Data/charlie_suit_and_wand_demo.csv"
    df = extractDataFrameFromCSV(dataLocation= dataLocation)

    print(df)


if __name__ == "__main__":
    testExtractDataFrameFromCSV()