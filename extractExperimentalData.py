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




# extract the experimental data onto a df, test file will check whether 
# rows skipped will need to be updated in the future
df = pd.read_csv(dataLocation,skiprows=[0,1,2,4],header = None)

# the top row has the names of each part so extract this
bodyParts = df.iloc[0].values
# extract the kinematic nature of each column (rotation or position)
kinematicType = df.iloc[1].values
# extract the variable in third row
kinematicVariable = df.iloc[2].values

# create a header array to store a simplified header for each column
headerArray = []
headerArray.append('Frame')
headerArray.append('Time (Seconds)')
wandIndex = None
otherIndex = None

for i in range(2,df.shape[1]):
  headerArray.append(bodyParts[i] + ' ' + kinematicType[i] + ' ' + kinematicVariable[i])
  if otherIndex == None and "Charlie" not in bodyParts[i]:
    otherIndex = i
  if wandIndex == None and "Wand"  in bodyParts[i]:
    wandIndex = i

# now create dataframe removing the previous rows of metadata and reassigning the
# column titles

df = df.iloc[3:]
df.columns = headerArray
print(df.head())
df = df.astype(float)

charlieMotiondf = df.loc[:,headerArray[2:otherIndex]]

wandMotiondf = df.loc[:,headerArray[wandIndex:-1]]

plt.plot(charlieMotiondf)
plt.title('Charlie body trackers')
plt.show()

plt.plot(wandMotiondf)
plt.title('Wand trackers')
plt.show()

# DATA ANALYSIS

# try to

print('Program successfully ran.')