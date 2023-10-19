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

        try:
            dataLocation = "Data/charlie_suit_and_wand_demo.csv"
            df = extractDataFrameFromCSV(dataLocation= dataLocation, includeCols='Charlie')
        except FileNotFoundError: # execute lines below if the file is being called from the directory it lies in
            try:
                dataLocation = "../Data/charlie_suit_and_wand_demo.csv"
                df = extractDataFrameFromCSV(dataLocation= dataLocation,includeCols='Charlie')
                print('')
            except FileNotFoundError:
                raise Exception("Unusual working directory discovered, current directory is: {}".format(os.getcwd()))

        assert df.shape == (1801,482)



if __name__ == "__main__":
    testExtractDataFrameFromCSV()