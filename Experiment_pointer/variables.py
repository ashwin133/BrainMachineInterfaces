"""
Stores all variables needed for pointer experiment
"""


# import necessary libraries
import numpy as np
import sys
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')

from Experiment_pointer.objects import *

#toggle screen size
worldx = 1100
worldy = 800

LATENCY_TEST = False

fps = 80 # frame rate
ani = 4 # animation cycles # animate simple movements repeatedly

# colours
BLUE = (25, 25, 200)
BLACK = (23, 23, 23)
WHITE = (254, 254, 254)
RED  = (255,0,0)
GREEN = (0,255,0)
colours = {'BLUE': BLUE , 'BLACK': BLACK, 'WHITE':WHITE, 'RED':RED, 'GREEN':GREEN

}

main = True

# box properties
leftCornerXBoxLoc = np.random.randint(100,1000)
leftCornerYBoxLoc = np.random.randint(100,700)
boxWidth = 60
boxHeight = 60
boxColor = RED

if LATENCY_TEST:
    boxWidth = worldx
    boxHeight = 80
    leftCornerXBoxLoc = 0
    leftCornerYBoxLoc = 300

debugMode = True
timeToReach = None

# DECIDE WHETHER TO READ ONLINE DATA AND WHETHER TO READ OR RECORD DATA
FETCHDATAFROMREALTIME = True
recordData = False
readData = False
readRigidBodies = True # turn off if not read data
readAdjustedRigidBodies = True # turn off if not read data 
showCursorPredictor = False #turn off if not read data 
readLocation = 'PointerExperimentData/23_11_ashTrial1_120s_test_usingModelDecoderB.npz'
writeDataLocation = 'PointerExperimentData/23_11_ashTrial1_120s_test_usingModelDecoderB3.npz'
writeDataLocationPkl = 'PointerExperimentData/23_11_ashTrial1_120s_test_usingModelDecoderB3.pkl'
metadataLocation = 'metadata'
invertXaxis = True  # necessary when facing opposite direction
metadata = {'MetaData:' 'Pres '
}
runDecoderInLoop = False
decoderType = "D"
retrieveCursorDataFromModelFile = False
modelReadLocation = 'PointerExperimentData/linearRigidBodyDModel.npz'



handDataReadVarName = 'dataStore'
targetBoxReadVarName = 'targetBoxLocs'
targetBoxTimeAppearsVarName = 'targetBoxAppearTimes'
targetBoxTimeHitsVarName = 'targetBoxHitTimes'
allBodyDataVarName = 'allBodyPartsData'
boxSizeVarName = 'boxSize'
cursorMotionDatastoreLocation = 'PointerExperimentData/23_11_ashTrial1_90s_linearPredCursorPos.npz' 
gameEngineLocation = 'GameEngine'
#time to run program
timeProgram = 120 # in seconds
testMode = False


reachedBoxStatus = 0
reachedBoxLatch = False

#calibration
calibrated = False


# record times taken to hit each box
boxHitTimes = []

enforce = True
offline = True
positions = True

processedRigidBodyParts = ['Pelvis', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 
                      'LFArm', 'LHand', 'RShoulder', 'RUArm', 'RFArm', 'RHand']


gameEngine = gameStatistics(worldx,worldy,LATENCY_TEST,fps,ani,colours,main,timeToReach,
                 FETCHDATAFROMREALTIME,recordData,readData,readLocation,writeDataLocation,
                 metadataLocation,metadata,handDataReadVarName,targetBoxReadVarName,
                 targetBoxTimeAppearsVarName,targetBoxTimeHitsVarName,allBodyDataVarName,
                 boxSizeVarName,timeProgram,reachedBoxStatus,reachedBoxLatch,calibrated,
                 boxHitTimes,enforce,offline,positions,processedRigidBodyParts,
                 leftCornerXBoxLoc,leftCornerYBoxLoc,boxWidth,boxHeight,testMode,readRigidBodies,
                 readAdjustedRigidBodies,showCursorPredictor,cursorMotionDatastoreLocation,runDecoderInLoop,
                 retrieveCursorDataFromModelFile = retrieveCursorDataFromModelFile,modelReadLocation = modelReadLocation,decoderType = decoderType,writeDataLocationPkl = writeDataLocationPkl,invertXaxis=invertXaxis)


