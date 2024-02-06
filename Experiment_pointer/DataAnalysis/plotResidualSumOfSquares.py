"""
This programs plots R squared values for different models applied to decode data  from the set of trials on the 23rd of November
"""
#23_11_ashTrial1_90s.npz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.decomposition import PCA
from sklearn import linear_model
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')
from lib_streamAndRenderDataWorkflows.config_streaming import renderingBodyParts

scores = []
scoresMetaData = []
scoresLabels =  []

def findR2Val(mode,tester,compPca,colorMap = None,plot = False,DOFOffset = 0.03):
    rigidBodies1, cursorPos1,cursorVel1,goCues1,targetHits1 = processTrialData('23_11_ashTrial1_90s.npz', '23_11_trial1_cal_matrix.npz',DOFOffset)# make this test as it is shorter
    rigidBodies2, cursorPos2,cursorVel2,goCues2,targetHits2 = processTrialData('23_11_ashTrial2_120s.npz', '23_11_trial2_cal_matrix.npz',DOFOffset)
    rigidBodies3, cursorPos3,cursorVel3,goCues3,targetHits3 = processTrialData('23_11_ashTrial3_120s.npz', '23_11_trial3_cal_matrix.npz',DOFOffset)
    rigidBodies4, cursorPos4,cursorVel4,goCues4,targetHits4 = processTrialData('23_11_ashTrial4_120s.npz', '23_11_trial4_cal_matrix.npz',DOFOffset)
    rigidBodies5, cursorPos5,cursorVel5,goCues5,targetHits5 = processTrialData('23_11_ashTrial5_120s.npz', '23_11_trial5_cal_matrix.npz',DOFOffset)

    rigidBodyVectorTraining = np.concatenate((rigidBodies2,rigidBodies3,rigidBodies4,rigidBodies5), axis = 0)
    cursorPosTraining = np.concatenate((cursorPos2,cursorPos3,cursorPos4,cursorPos5),axis = 0)

    rigidBodyVectorTest = rigidBodies1
    cursorPosTest = cursorPos1

    if mode == 'RigidBodiesSetA':
        # delete right hand only SET A
        type = 'A'
        idxRightHand = renderingBodyParts.index('RHand') * 6
        X_train = np.delete(rigidBodyVectorTraining,slice(idxRightHand,idxRightHand+6,1),1)
        X_test_linear = np.delete(rigidBodyVectorTest,slice(idxRightHand,idxRightHand+6,1),1)
        X_test_ridge = np.delete(rigidBodyVectorTest,slice(idxRightHand,idxRightHand+6,1),1)

    elif mode == 'RigidBodiesSetB':
        # delete right side rigid bodies, SET B
        type = 'B'
        idxRightHand = renderingBodyParts.index('RHand') * 6
        idxRightShoulder = renderingBodyParts.index('RShoulder') * 6
        X_train = np.delete(rigidBodyVectorTraining,slice(idxRightShoulder,idxRightHand+6,1),1)
        X_test_linear = np.delete(rigidBodyVectorTest,slice(idxRightShoulder,idxRightHand+6,1),1)
        X_test_ridge = np.delete(rigidBodyVectorTest,slice(idxRightShoulder,idxRightHand+6,1),1)
    
    elif mode == 'RigidBodiesSetC':
        # # only get the left hand
        type = 'C'
        idxLeftHand = renderingBodyParts.index('LHand') * 6
        X_train = rigidBodyVectorTraining[:,idxLeftHand:idxLeftHand+6]
        X_test_linear = rigidBodyVectorTest[:,idxLeftHand:idxLeftHand+6]
        X_test_ridge = rigidBodyVectorTest[:,idxLeftHand:idxLeftHand+6]
    
    elif mode == 'RigidBodiesSetD':
        # # # only get the right hand
        type = 'D'
        idxRightHand = renderingBodyParts.index('RHand') * 6
        X_train = rigidBodyVectorTraining[:,idxRightHand:idxRightHand+6]
        X_test_linear = rigidBodyVectorTest[:,idxRightHand:idxRightHand+6]
        X_test_ridge = rigidBodyVectorTest[:,idxRightHand:idxRightHand+6]
    
    if tester == 'PCA_linear':
        pca = PCA(n_components=compPca)
        pca.fit(X_train)
        pcaRes = pca.explained_variance_ratio_
        np.set_printoptions(suppress=True)
        #print(pcaRes)
        components = pca.components_
        #print(pca.components_)
        cumSumVar = np.cumsum(pcaRes)
        # plt.plot(cumSumVar)
        # plt.show()
        # transform all matrices to lower dimension
        X_train_pca = np.matmul(components,X_train.transpose()).transpose()
        Y_train = cursorPosTraining
        reg  = linear_model.LinearRegression().fit(X_train_pca, Y_train)
        # predict for trial 1
        X_test_linear_pca = np.matmul(components,X_test_linear.transpose()).transpose()
        Y_test_linear = reg.predict(X_test_linear_pca)
        score = reg.score(X_test_linear_pca,cursorPos1)
        print('Score:' ,score)
        Y_pred = Y_test_linear
    
    elif tester == 'linear':
        Y_train = cursorPosTraining
        reg  = linear_model.LinearRegression().fit(X_train, Y_train)
        # predict for trial 1
        Y_test_linear = reg.predict(X_test_linear)
        score = reg.score(X_test_linear,cursorPos1)
        print('Score:' , score)
        Y_pred = Y_test_linear


        

    if plot:
        correctY = cursorPosTest
        for i in range(0,len(colorMap)): # len(goCues1)
            plotFrom = goCues1[i]
            plotUntil = targetHits1[i]
            plt.plot(correctY[plotFrom:plotUntil,0],correctY[plotFrom:plotUntil,1],color = colorMap[i])
            plt.plot(Y_pred[plotFrom:plotUntil,0],Y_pred[plotFrom:plotUntil,1], color = colorMap[i])
            if i == 0:
                plt.scatter(correctY[plotFrom,0], correctY[plotFrom,1],s=250, marker=".", color = 'g',label = 'Actual cursor start position')
                # plt.scatter(correctY[plotUntil,0], correctY[plotUntil,1], s=100, marker="D", color = 'g')
                plt.scatter(Y_pred[plotFrom,0], Y_pred[plotFrom,1],s=250, marker=".", color = 'r',label = 'Estimated cursor start position')
                # plt.scatter(Y_pred[plotUntil,0], Y_pred[plotUntil,1], s=60, marker="D", color = 'r')
            else:
                plt.scatter(correctY[plotFrom,0], correctY[plotFrom,1],s=250, marker=".", color = 'g')
                # plt.scatter(correctY[plotUntil,0], correctY[plotUntil,1], s=100, marker="D", color = 'g')
                plt.scatter(Y_pred[plotFrom,0], Y_pred[plotFrom,1],s=250, marker=".", color = 'r')
                # plt.scatter(Y_pred[plotUntil,0], Y_pred[plotUntil,1], s=60, marker="D", color = 'r')
        
        plt.xlabel('Normalised X pos on game screen',fontsize = 15)
        plt.ylabel('Normalised Y pos on game screen', fontsize = 15)
        plt.title('Trajectories showing actual and estimated cursor position for each target aquisition performed in test set. \n Each trajectory is shown in a different colour and position estimates are derived from set (b) of rigid bodies',fontsize = 15)
        plt.legend()
        plt.show()
    if compPca is not None: 
        scoreLabel =  'l_PCA:' +str(compPca) + ', ' + str(DOFOffset)
    else:
        scoreLabel = 'l' +  ', ' + str(DOFOffset)

    return score,scoreLabel


    
        
        
def processTrialData(dataLocation,calLocation,DOFOffset = 0.03):
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

    # now get times of when target appeared to when target was hit
    targetBoxHitTimes = np.array(data['targetBoxHitTimes'])
    targetBoxAppearTimes = np.array(data['targetBoxAppearTimes'])
    # get the relevant elements of targetBoxAppearTimes
    zeroIdx = np.where(targetBoxAppearTimes == 0)[0][0]
    targetBoxAppearTimes = targetBoxAppearTimes[0:zeroIdx]
    goCueIdxes = [np.argmin(np.abs(cursorMotion[:,0] - a)) for a in targetBoxAppearTimes]
    targetAquiredIdxes = [np.argmin(np.abs(cursorMotion[:,0] - a)) for a in targetBoxHitTimes]

    cursorMotion_noTimestamp = cursorMotion[:,1:] # remove timestamp column
    simpleBodyParts = [0,1,2,3,4,5,6,7,8,24,25,26,27,43,44,45,47,48,49]
    rigidBodyData  = rigidBodyData.reshape(-1,51,6)
    rigidBodyData = rigidBodyData[:,simpleBodyParts,:].reshape(-1,114)
    noDOF = 114
    if True:
        for DOF in range(0,noDOF):
            DOFMin = min(rigidBodyData[:,DOF])
            DOFMax = max(rigidBodyData[:,DOF])
            rigidBodyData[:,DOF] =  (rigidBodyData[:,DOF] - DOFMin) / (DOFMax - DOFMin + DOFOffset) # very sensitive to the offset ???

        cursorDOF = 2
        for cursorDim in range(0,cursorDOF):
            cursorDOFmin = min(cursorMotion_noTimestamp[:,cursorDim])
            if False: # make min and max x,y cursor pos the actual range set in pygame
                if cursorDim == 0:
                    cursorDOFmin = 0
                    cursorDOFMax = 1100
                else:
                    cursorDOFmin = 0
                    cursorDOFmax = 800
            cursorDOFmax = max(cursorMotion_noTimestamp[:,cursorDim])

            cursorMotion_noTimestamp[:,cursorDim] = (cursorMotion_noTimestamp[:,cursorDim] - cursorDOFmin) / (cursorDOFmax - cursorDOFmin+ 5)

    # def plotCursorMotion(cursorMotion):
    #     ax = plt.figure()
    #     plt.plot(cursorMotion[0:400,1],-cursorMotion[0:400,2])
    #     plt.show()
    # #plotCursorMotion(cursorMotion)

    
    return rigidBodyData, cursorMotion_noTimestamp,cursorVelocities,goCueIdxes,targetAquiredIdxes



colorMap =  [
    'red',         # Standard named color
    '#FFA07A',     # Light Salmon (hexadecimal)
    'blue',        # Standard named color
    '#00FA9A',     # Medium Spring Green (hexadecimal)
    'green',       # Standard named color
    '#FFD700',     # Gold (hexadecimal)
    'purple',      # Standard named color
    '#87CEFA',     # Light Sky Blue (hexadecimal)
    'orange',      # Standard named color
    '#FF69B4',     # Hot Pink (hexadecimal)
    'cyan',        # Standard named color
    '#8A2BE2',     # Blue Violet (hexadecimal)
    'magenta',     # Standard named color
    '#20B2AA',     # Light Sea Green (hexadecimal)
    'brown',       # Standard named color
    '#D2691E',     # Chocolate (hexadecimal)
    'pink',        # Standard named color
    '#6495ED'      # Cornflower Blue (hexadecimal)
]

# 
scoresA = []
scoresLabelsA = []

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetA',tester = 'linear',compPca = None, colorMap=colorMap,plot=False,DOFOffset= 0.01)
scoresA.append(score1)
scoresLabelsA.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetA',tester = 'linear',compPca = None, colorMap=colorMap,plot=False,DOFOffset= 0.03)
scoresA.append(score1)
scoresLabelsA.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetA',tester = 'linear',compPca = None, colorMap=colorMap,plot=False,DOFOffset= 0.06)
scoresA.append(score1)
scoresLabelsA.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetA',tester = 'PCA_linear',compPca = 20, colorMap=colorMap,plot=False,DOFOffset= 0.03)
scoresA.append(score1)
scoresLabelsA.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetA',tester = 'PCA_linear',compPca = 40, colorMap=colorMap,plot=False,DOFOffset= 0.03)
scoresA.append(score1)
scoresLabelsA.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetA',tester = 'PCA_linear',compPca = 60, colorMap=colorMap,plot=False,DOFOffset= 0.03)
scoresA.append(score1)
scoresLabelsA.append(scoreLabel1)

plt.bar(scoresLabelsA,scoresA,width = 0.5)
plt.ylabel('Residual sum of squares value',fontsize = 15)
plt.xlabel('Model Configurations',fontsize = 15)
plt.title("Plot to show how different model configurations affect the accuracy of the model's predictions \n - For using rigid body set A (All rigid bodies except right hand)",fontsize = 15)
plt.savefig('Plot_rigidBodiesA_Rsquare.png')
plt.show()

scoresB = []
scoresLabelsB = []

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetB',tester = 'PCA_linear',compPca = 20, colorMap=colorMap,plot=False,DOFOffset= 0.03)
scoresB.append(score1)
scoresLabelsB.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetB',tester = 'PCA_linear',compPca = 40, colorMap=colorMap,plot=False,DOFOffset= 0.03)
scoresB.append(score1)
scoresLabelsB.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetB',tester = 'PCA_linear',compPca = 60, colorMap=colorMap,plot=False,DOFOffset= 0.03)
scoresB.append(score1)
scoresLabelsB.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetB',tester = 'linear',compPca = None, colorMap=colorMap,plot=False,DOFOffset= 0.05)
scoresB.append(score1)
scoresLabelsB.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetB',tester = 'linear',compPca = None, colorMap=colorMap,plot=False,DOFOffset= 0.03)
scoresB.append(score1)
scoresLabelsB.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetB',tester = 'linear',compPca = None, colorMap=colorMap,plot=False,DOFOffset= 0.01)
scoresB.append(score1)
scoresLabelsB.append(scoreLabel1)

plt.bar(scoresLabelsB,scoresB,width = 0.5)
plt.ylabel('Residual sum of squares value',fontsize = 15)
plt.xlabel('Model Configurations',fontsize = 15)
plt.title("Plot to show how different model configurations affect the accuracy of the model's predictions \n - For using rigid body set B (All rigid bodies except ones on right arm)",fontsize = 15)
plt.savefig('Plot_rigidBodiesB_Rsquare.png')
plt.show()


scoresC = []
scoresLabelsC = []

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetC',tester = 'linear',compPca = None, colorMap=colorMap,plot=False,DOFOffset= 0.03)
scoresC.append(score1)
scoresLabelsC.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetC',tester = 'linear',compPca = None, colorMap=colorMap,plot=False,DOFOffset= 0.05)
scoresC.append(score1)
scoresLabelsC.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetC',tester = 'linear',compPca = None, colorMap=colorMap,plot=False,DOFOffset= 0.1)
scoresC.append(score1)
scoresLabelsC.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetC',tester = 'PCA_linear',compPca = 3, colorMap=colorMap,plot=False,DOFOffset= 0.03)
scoresC.append(score1)
scoresLabelsC.append(scoreLabel1)

plt.bar(scoresLabelsC,scoresC,width = 0.5)
plt.ylabel('Residual sum of squares value',fontsize = 15)
plt.xlabel('Model Configurations',fontsize = 15)
plt.title("Plot to show how different model configurations affect the accuracy of the model's predictions \n - For using rigid body set C (left hand only)",fontsize = 15)
plt.savefig('Plot_rigidBodiesC_Rsquare.png')
plt.show()



scoresD = []
scoresLabelsD = []

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetD',tester = 'linear',compPca = None, colorMap=colorMap,plot=False,DOFOffset= 0.03)
scoresD.append(score1)
scoresLabelsD.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetD',tester = 'linear',compPca = None, colorMap=colorMap,plot=False,DOFOffset= 0.1)
scoresD.append(score1)
scoresLabelsD.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetD',tester = 'linear',compPca = None, colorMap=colorMap,plot=False,DOFOffset= 0.3)
scoresD.append(score1)
scoresLabelsD.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetD',tester = 'PCA_linear',compPca = 2, colorMap=colorMap,plot=False,DOFOffset= 0.03)
scoresD.append(score1)
scoresLabelsD.append(scoreLabel1)

score1,scoreLabel1 = findR2Val(mode = 'RigidBodiesSetD',tester = 'PCA_linear',compPca = 2, colorMap=colorMap,plot=False,DOFOffset= 0.06)
scoresD.append(score1)
scoresLabelsD.append(scoreLabel1)

plt.bar(scoresLabelsD,scoresD,width = 0.5)
plt.ylabel('Residual sum of squares value',fontsize = 15)
plt.xlabel('Model Configurations',fontsize = 15)
plt.title("Plot to show how different model configurations affect the accuracy of the model's predictions\n - For using rigid body set D (right hand only [control])",fontsize = 15)

plt.savefig('Plot_rigidBodiesD_Rsquare.png')
plt.show()
print('Program ran successfully')

def calcThetha(y_curr,y_prev,x_curr,x_prev):
    dY = y_curr - y_prev
    dX = x_curr - x_prev
    alpha = np.abs(np.arctan((dY)/(dX)))
    if dY >= 0 and dX >= 0:
        return alpha
    elif dY >= 0 and dX <= 0:
        return np.pi - alpha
    elif dY <= 0 and dX >= 0:
        return -alpha
    elif dY <= 0 and dX <= 0:
        return  np.pi + alpha

def calcDirAvgForMovingToTarget(cursorPos):
    thethas = np.zeros(len(cursorPos[:,0]) - 1)
    for i in range(1,len(cursorPos[:,0])):
        thethas[i] = calcThetha(cursorPos[i,1],cursorPos[i-1,1],cursorPos[i,0],cursorPos[i-1,0])
    return np.average(thethas)

def reportThethaErrorForEachTargetMove(targetMotionCursorPosTrue,targetMotionCursorPosEst):
    thethaAvgTrue = calcDirAvgForMovingToTarget(targetMotionCursorPosTrue)
    thethaAvgEst = calcDirAvgForMovingToTarget(targetMotionCursorPosEst)
    return thethaAvgTrue - thethaAvgEst

def feedTargetMotionCursorPos(trueTrialCursorPos,estTrialCursorPos,goCueIdxes,targetAquiredIdxes):
    trialDirDifferences = np.zeros(len(targetAquiredIdxes))
    for i in range(len(targetAquiredIdxes)):
        cursorPosTargetMotionTrue = trueTrialCursorPos[goCueIdxes[i]: targetAquiredIdxes[i],:]
        cursorPosTargetMotionEst = estTrialCursorPos[goCueIdxes[i]: targetAquiredIdxes[i], :]
        trialDirDifferences[i] = reportThethaErrorForEachTargetMove(cursorPosTargetMotionTrue,cursorPosTargetMotionEst)
    return trialDirDifferences
