"""
Contains tests for workflows involved in streaming data from motive and related to shared memory
"""
# import standard python libraries
import sys
import pytest
import warnings
import pytest
import os

# add Root Directory to system path to import created packages
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')

# import created packages
from StreamAndRenderDataWorkflows.streamData import extractDataFrameFromCSV



def testExtractDataFrameFromCSV():
    with pytest.warns(UserWarning):
        warnings.warn("DtypeWarning", UserWarning) # added as currently a warning about datatypes exist when importing csv

        if os.getcwd() == '/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces':
            dataLocation = "Data/charlie_suit_and_wand_demo.csv"
            df = extractDataFrameFromCSV(dataLocation= dataLocation)
        elif os.getcwd() == '/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces/tests_StreamAndRenderDataWorkflows':
            dataLocation = "../Data/charlie_suit_and_wand_demo.csv"
            df = extractDataFrameFromCSV(dataLocation= dataLocation)
        else:
            raise Exception("Unusual working directory discovered, current directory is: {}".format(os.getcwd()))


    #print(df)


if __name__ == "__main__":
    testExtractDataFrameFromCSV()