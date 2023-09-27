"""
This reads a sample demo file where the suit and the wand is used.

@Ashwin Gunasekaran 26-09-23
"""

# import all dependencies

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import openpyxl
import experimentalDataVariables as expVars

# import csv data into a table
dataLocation = expVars.fileLocation

# create a list to hold all metadata
metadata = []

with open(dataLocation, 'r') as file:
  csvreader = csv.reader(file, delimiter=':')
  i = 0
  for idx,row in enumerate(csvreader):
    if i > 5: # currently only 6 rows of metadata, there is a test to ensure this
      break
    i += 1
    metadata.append(row)

#print(metadata)


# extract the experimental data onto a df, test file will check whether 
# rows skipped will need to be updated in the future
df = pd.read_csv(dataLocation,skiprows=[0,1,2,3,4,5])

print(df.head())
print('Program successfully ran.')