"""
This programs covers workflows to create a dataset from the set of trials on the 23rd of November
"""
#23_11_ashTrial1_90s.npz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def processTrialData(dataLocation,calLocation):
    try:
        data = np.load('../PointerExperimentData/' + dataLocation) # for siddhi trial 3 the boxes were 60 x 60
        calMatrix = np.load('../PointerExperimentData/' + calLocation)
    except FileNotFoundError:
        data = np.load('Experiment_pointer/PointerExperimentData/' + dataLocation)
        calMatrix = np.load('Experiment_pointer/PointerExperimentData/' + calLocation)

    # data starts as soon as cursor moves on screen
    # recieve list of cursor movements
    cursorMotion = data['cursorMotionDatastoreLocation']    
    # recieve list of transformed rigid body vectors that correspond to cursor movements
    calMatrix = calMatrix['calMatrix']
    rigidBodyData_trial1 = data['allBodyPartsData'] # raw motion of all rigid bodies
    rigidBodyData_trial1 = rigidBodyData_trial1.reshape(-1,51,6)
    rigidBodyData_normalised = np.tensordot(calMatrix,rigidBodyData_trial1.transpose(), axes=([1],[0])).transpose().reshape(-1,306)

    # find when data stops being recorded for cursor data
    lastrecordCursorIdx = np.where(cursorMotion[:,0] == 0)[0][0] - 1
    lastrecordRigidBodyIdx = np.where(rigidBodyData_normalised[:,0] == 0)[0][0] - 1
    startRigidBodyIdx = lastrecordRigidBodyIdx - lastrecordCursorIdx # as this is when calibration finishes and the cursor starts to move

    rigidBodyData = rigidBodyData_normalised[startRigidBodyIdx:lastrecordRigidBodyIdx+1,:]
    cursorMotion = cursorMotion[0:lastrecordCursorIdx+1]
    cursorMotion_noTimestamp = cursorMotion[:,1:] # remove timestamp column
    simpleBodyParts = [0,1,2,3,4,5,6,7,8,24,25,26,43,44,45,47,48,49]
    rigidBodyData  = rigidBodyData.reshape(-1,51,6)
    rigidBodyData = rigidBodyData[:,simpleBodyParts,:].reshape(-1,108)

  


    # def plotCursorMotion(cursorMotion):
    #     ax = plt.figure()
    #     plt.plot(cursorMotion[0:400,1],-cursorMotion[0:400,2])
    #     plt.show()
    # #plotCursorMotion(cursorMotion)


    
    return rigidBodyData, cursorMotion_noTimestamp


rigidBodies1, cursorPos1 = processTrialData('23_11_ashTrial1_90s.npz', '23_11_trial1_cal_matrix.npz')# make this test as it is shorter
rigidBodies2, cursorPos2 = processTrialData('23_11_ashTrial2_120s.npz', '23_11_trial2_cal_matrix.npz')
rigidBodies3, cursorPos3 = processTrialData('23_11_ashTrial3_120s.npz', '23_11_trial3_cal_matrix.npz')
rigidBodies4, cursorPos4 = processTrialData('23_11_ashTrial4_120s.npz', '23_11_trial4_cal_matrix.npz')
rigidBodies5, cursorPos5 = processTrialData('23_11_ashTrial5_120s.npz', '23_11_trial5_cal_matrix.npz')



  # now we need to pre process the data for analyis
# first normalise each deg of freedom for range, there are 114


noDOF = 114
# find max and min values for each DOF and rescale
DOFMaxValues = []
rigidBodyVector = rigidBodies2
if False:
    for DOF in range(0,noDOF):
        DOFMaxValues.append(max(rigidBodyVector[:,DOF]))
        DOFMin = min(rigidBodyVector[:,DOF])
        DOFMax = max(rigidBodyVector[:,DOF])
        rigidBodyVector[:,DOF] =  (rigidBodyVector[:,DOF] - DOFMin) / (DOFMax - DOFMin + 0.03)

rigidBodies2_right_hand = rigidBodies2[:,:]

from sklearn import linear_model
X = rigidBodies2_right_hand
Y = cursorPos2
reg  = linear_model.LinearRegression().fit(X, Y)

# # now perform pca to attempt to reduce dimensionality
# from sklearn.decomposition import PCA
# pca = PCA(n_components=114)
# pca.fit(rigidBodyVector)
# pcaRes = pca.explained_variance_ratio_
np.set_printoptions(suppress=True)
# print(pcaRes)
# components = pca.components_
# print(pca.components_)

# then subtract mean
print('Program ran successfully')