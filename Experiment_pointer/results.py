"""
a program to visualise the results from a trial
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
try:
    data = np.load('PointerExperimentData/23_11_ashTrial1.npz') # for siddhi trial 3 the boxes were 60 x 60
except FileNotFoundError:
    data = np.load('Experiment_pointer/PointerExperimentData/23_11_ashTrial1.npz')

targetBoxAppearTimes = np.array(data['targetBoxAppearTimes'])
targetBoxHitTimes = np.array(data['targetBoxHitTimes'])
print('Target Box appear times:', targetBoxAppearTimes)
print('Target Box Hit times:', targetBoxHitTimes)
# get the relevant elements of targetBoxAppearTimes
zeroIdx = np.where(targetBoxAppearTimes == 0)[0][0]
targetBoxAppearTimes = targetBoxAppearTimes[0:zeroIdx]


df = data['allBodyPartsData']
df = pd.DataFrame(df)
df.to_csv('presentation_demo_rigidBodies.csv')


#plt times at which target box was hit
# plt.scatter(targetBoxAppearTimes,np.ones(len(targetBoxAppearTimes)))
# plt.scatter(targetBoxHitTimes,np.ones(len(targetBoxHitTimes))+0.1)
# plt.ylim(0,1.2)
# plt.xlim(0,targetBoxHitTimes[-1]*1.1)
# plt.show()


# reaction times
reactionTimes = targetBoxHitTimes - targetBoxAppearTimes
plt.hist(reactionTimes/1000,bins = 8,range = (0,0.8))
plt.xlabel('Reaction Time (seconds)')

print('reaction times', reactionTimes)
print('Average Reaction Time Ashwin: ',np.average(reactionTimes))

# try:
#     dataSid = np.load('siddhiTrial3.npz') # for siddhi trial 3 the boxes were 60 x 60
# except FileNotFoundError:
#     dataSid = np.load('Experiment_pointer/siddhiTrial3.npz')

# targetBoxAppearTimesSid = np.array(dataSid['targetBoxAppearTimes'])
# targetBoxHitTimesSid = np.array(dataSid['targetBoxHitTimes'])

# # get the relevant elements of targetBoxAppearTimes
# zeroIdxSid = np.where(targetBoxAppearTimesSid == 0)[0][0]
# targetBoxAppearTimesSid = targetBoxAppearTimesSid[0:zeroIdxSid]
# reactionTimesSid = targetBoxHitTimesSid - targetBoxAppearTimesSid
# plt.hist((reactionTimesSid/1000)+1,bins = 9,range = (1,1.8))
# plt.show()

# print('Average Reaction Time Siddhi: ',np.average(reactionTimesSid))




