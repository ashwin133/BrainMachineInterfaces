"""
This programs covers workflows to create a dataset from the set of trials on the 23rd of November
"""
#23_11_ashTrial1_90s.npz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')
from lib_streamAndRenderDataWorkflows.config_streaming import renderingBodyParts

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
    
    cursorVelocities = np.gradient(cursorMotion[:,1:],cursorMotion[:,0],axis=0)
    cursorMotion_noTimestamp = cursorMotion[:,1:] # remove timestamp column
    simpleBodyParts = [0,1,2,3,4,5,6,7,8,24,25,26,27,43,44,45,47,48,49]
    rigidBodyData  = rigidBodyData.reshape(-1,51,6)
    rigidBodyData = rigidBodyData[:,simpleBodyParts,:].reshape(-1,114)
    noDOF = 114
    if True:
        for DOF in range(0,noDOF):
            DOFMin = min(rigidBodyData[:,DOF])
            DOFMax = max(rigidBodyData[:,DOF])
            rigidBodyData[:,DOF] =  (rigidBodyData[:,DOF] - DOFMin) / (DOFMax - DOFMin + 0.03)

        cursorDOF = 2
        for cursorDim in range(0,cursorDOF):
            cursorDOFmin = min(cursorMotion_noTimestamp[:,cursorDim])
            cursorDOFmax = max(cursorMotion_noTimestamp[:,cursorDim])
            cursorMotion_noTimestamp[:,cursorDim] = (cursorMotion_noTimestamp[:,cursorDim] - cursorDOFmin) / (cursorDOFmax - cursorDOFmin + 5)

    # def plotCursorMotion(cursorMotion):
    #     ax = plt.figure()
    #     plt.plot(cursorMotion[0:400,1],-cursorMotion[0:400,2])
    #     plt.show()
    # #plotCursorMotion(cursorMotion)

    
    return rigidBodyData, cursorMotion_noTimestamp,cursorVelocities


rigidBodies1, cursorPos1,cursorVel1 = processTrialData('23_11_ashTrial1_90s.npz', '23_11_trial1_cal_matrix.npz')# make this test as it is shorter
rigidBodies2, cursorPos2,cursorVel2 = processTrialData('23_11_ashTrial2_120s.npz', '23_11_trial2_cal_matrix.npz')
rigidBodies3, cursorPos3,cursorVel3 = processTrialData('23_11_ashTrial3_120s.npz', '23_11_trial3_cal_matrix.npz')
rigidBodies4, cursorPos4,cursorVel4 = processTrialData('23_11_ashTrial4_120s.npz', '23_11_trial4_cal_matrix.npz')
rigidBodies5, cursorPos5,cursorVel5 = processTrialData('23_11_ashTrial5_120s.npz', '23_11_trial5_cal_matrix.npz')

rigidBodyVector = np.concatenate((rigidBodies2,rigidBodies3,rigidBodies4,rigidBodies5), axis = 0)
cursorPosTraining = np.concatenate((cursorPos2,cursorPos3,cursorPos4,cursorPos5),axis = 0)
  # now we need to pre process the data for analyis
# first normalise each deg of freedom for range, there are 114

# delete all right data
idxRightHand = renderingBodyParts.index('RHand') * 6
idxRightShoulder = renderingBodyParts.index('RShoulder') * 6
X = np.delete(rigidBodyVector,slice(idxRightShoulder,idxRightHand+6,1),1)
X_pred_linear = np.delete(rigidBodies1,slice(idxRightShoulder,idxRightHand+6,1),1)
X_pred_ridge = np.delete(rigidBodies1,slice(idxRightShoulder,idxRightHand+6,1),1)

# # only get the left hand
# idxLeftHand = renderingBodyParts.index('LHand') * 6
# rigidBodies2_left_hand = rigidBodyVector[:,idxLeftHand:idxLeftHand+6]
# X = rigidBodies2_left_hand
# X_pred_linear = rigidBodies1[:,idxLeftHand:idxLeftHand+6]
# X_pred_ridge = rigidBodies1[:,idxLeftHand:idxLeftHand+6]


# # now perform pca to attempt to reduce dimensionality
from sklearn.decomposition import PCA
pca = PCA(n_components=40)
pca.fit(X)
pcaRes = pca.explained_variance_ratio_
np.set_printoptions(suppress=True)
print(pcaRes)
components = pca.components_
print(pca.components_)
cumSumVar = np.cumsum(pcaRes)
# plt.plot(cumSumVar)
# plt.show()




# # # only get the right hand
# idxRightHand = renderingBodyParts.index('RHand') * 6
# rigidBodies2_Right_hand = rigidBodies2[:,idxRightHand:idxRightHand+6]
# X = rigidBodies2_Right_hand

# delete the right hand column
# idxRightHand = renderingBodyParts.index('RHand') * 6
# X = np.delete(rigidBodies2,slice(idxRightHand,idxRightHand+6,1),1)



# transform all matrices to lower dimension
X_pca = np.matmul(components,X.transpose()).transpose()


# fit simple linear model to rigid body vector
from sklearn import linear_model
Y = cursorPosTraining
reg  = linear_model.LinearRegression().fit(X, Y)
XposCoeff = reg.coef_[0]
# predict for trial 1

X_pred_linear_pca = np.matmul(components,X_pred_linear.transpose()).transpose()
Y_pred_linear = reg.predict(X_pred_linear_pca)
print('score:', reg.score(X_pred_linear_pca, cursorPos1))

# fit ridge model to penalise high weights
from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(X_pca,Y)

# predict

X_pred_ridge_pca = np.matmul(components,X_pred_ridge.transpose()).transpose()
Y_pred_ridge = clf.predict(X_pred_ridge_pca)


Y_pred = Y_pred_linear
np.savez('Experiment_pointer/PointerExperimentData/23_11_ashTrial1_90s_linearPredCursorPos_rightonly.npz', cursorPred = Y_pred)

plotFrom = 600
plotUntil = 800
correctY = cursorPos1
plt.plot(correctY[plotFrom:plotUntil,0],correctY[plotFrom:plotUntil,1],label = 'correct')
plt.plot(Y_pred[plotFrom:plotUntil,0],Y_pred[plotFrom:plotUntil,1],label = 'predicted')

plt.legend()
# plt.show()


# then subtract mean
print('Program ran successfully')


