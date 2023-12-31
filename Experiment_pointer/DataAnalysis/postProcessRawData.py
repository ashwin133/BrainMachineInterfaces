"""
Contains some helper functions to post process the raw data so models can be applied

"""
import numpy as np

def processTrialData(dataLocation,calLocation,DOFOffset = 0.03,returnAsDict = False):
    """
    This function reads the raw trial data format that is generated after the pointer game is run. It processes 
    the data to extract useful information to fit models
    INPUT:
    @param dataLocation: location of raw trial data. Ensure file is inside PointerExperimentData, only supply the file name
    @param calLocation: location of file which has calibration matrix stored.  Ensure file is inside PointerExperimentData, only supply the file name
    @param DOFOffset: This adds the specified offset to each DOF's range
    @param returnAsDict: return data in a dictionary or as variables
    RETURNS:
    @param rigidBodyData: all rigid body movements across whole trial
    @param cursorMotion_noTimestamp: cursor motion across whole trial in form x,y
    @param cursorVelocities: cursor velocities across whole trial
    @param goCuesIdx: indexes into trial data corresponding to when target displayed on screen
    @param targetAquiredIdxes: index into trial data corresponding to when target reached
    @param timeStamps: timestamps for each index in trial data
    @param minDOF: minimum values for each DOF before scaling
    @param maxDOF: maximum values for each DOF before scaling
    """
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
    timeStamps = cursorMotion[:,0]
    simpleBodyParts = [0,1,2,3,4,5,6,7,8,24,25,26,27,43,44,45,47,48,49]
    rigidBodyData  = rigidBodyData.reshape(-1,51,6)
    rigidBodyData = rigidBodyData[:,simpleBodyParts,:].reshape(-1,114)
    noDOF = 114
    maxDOF = np.zeros(114)
    minDOF = np.zeros(114)

    if True:
        for DOF in range(0,noDOF):
            DOFMin = min(rigidBodyData[:,DOF])
            minDOF[DOF] = DOFMin
            DOFMax = max(rigidBodyData[:,DOF])
            maxDOF[DOF] = DOFMax
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



    if returnAsDict is not True:
        return rigidBodyData, cursorMotion_noTimestamp,cursorVelocities,np.array(goCueIdxes),np.array(targetAquiredIdxes), timeStamps,minDOF,maxDOF
    else:
        returnDict = {
            'rigidBodyData': rigidBodyData,
            'cursorPos': cursorMotion_noTimestamp,
            'cursorVel': cursorVelocities,
            'goCues': np.array(goCueIdxes),
            'targetReached': np.array(targetAquiredIdxes),
            'timestamps': timeStamps,
            'minDOF': minDOF,
            'maxDOF': maxDOF
        }
        return returnDict
    
def processMulipleTrialData(dataLocations,calLocation, DOFOffset =0.03):
    """
    Collates multiple trial data information together
    """

    trialDict = readIndividualTargetMovements(processTrialData(dataLocations[0],calLocation,DOFOffset,returnAsDict=True))
    del dataLocations[0]
    for file in dataLocations:
        trialDict_ = readIndividualTargetMovements(processTrialData(file,calLocation,DOFOffset,returnAsDict=True))
        for key in trialDict.keys():

            trialDict[key] = trialDict_[key] + trialDict[key]
    return trialDict
        
def readIndividualTargetMovements(processedDataDict):
    returnDict = {
        'rigidBodyData': [], 
        'cursorPosData': [], 
        'cursorVelData': [],
        'timestamps': []
    }
    for i in range(0,len(processedDataDict['goCues'])):
        startTime = processedDataDict['timestamps'][processedDataDict['goCues'][i]]
        returnDict['rigidBodyData'].append(processedDataDict['rigidBodyData'][processedDataDict['goCues'][i]:processedDataDict['targetReached'][i],:].transpose())
        returnDict['cursorPosData'].append(processedDataDict['cursorPos'][processedDataDict['goCues'][i]:processedDataDict['targetReached'][i],:].transpose())
        returnDict['cursorVelData'].append(processedDataDict['cursorVel'][processedDataDict['goCues'][i]:processedDataDict['targetReached'][i],:].transpose())
        returnDict['timestamps'].append(processedDataDict['timestamps'][processedDataDict['goCues'][i]:processedDataDict['targetReached'][i]] - startTime)    

    return returnDict