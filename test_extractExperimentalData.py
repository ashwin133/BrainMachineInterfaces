"""
test basic workflows
"""

# gather dependencies
import numpy as np
import sys
import experimentalDataVariables as expVars
import pandas as pd
import csv
dataLocation = expVars.fileLocation

def testNumpyImported():
    assert "numpy" in sys.modules

def testExperimentalMetaDataFormat():
    # check that the metadata is as expected
    metadata = []

    # first read all the metadata using csvreader
    with open(dataLocation, 'r') as file:
        csvreader = csv.reader(file, delimiter=':')
        i = 0
        for idx, row in enumerate(csvreader):
            if i > 5:  # currently only 6 rows of metadata, there is a test to ensure this
                break
            i += 1
            metadata.append(row)

    # next let's read the first row and test that it contains what we expect
    row1Data = metadata[0][0].split(',')
    assert row1Data[0] == 'Format Version'
    assert len(row1Data) == 24

    # let's repeat the same for the second row
    assert metadata[1] == []

    # repeat for third row
    row3Data = metadata[2][0].split(',')
    assert row3Data[1] == 'Type'

    # repeat for fourth row
    row4Data = metadata[3][0].split(',')
    assert row4Data[1] == 'Name'

    # repeat for fifth row
    row5Data = metadata[4][0].split(',')
    assert row5Data[1] == 'ID'

    # repeat for sixth row
    row6Data = metadata[5][0].split(',')
    assert row6Data[2] == 'Rotation'


def testExperimentalDataFormat():
    # check that the df ignores the correct rows of metadata
    df = pd.read_csv(dataLocation, skiprows=[0,1,2,3,4,5])
    assert df.columns[0] == 'Frame'
    assert df.columns[1] == 'Time (Seconds)'
