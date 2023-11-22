"""
Stores all variables needed for pointer experiment
"""

# import necessary libraries
import numpy as np

#toggle screen size
worldx = 1100
worldy = 800

LATENCY_TEST = False

fps = 100 # frame rate
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
readData = True
readLocation = 'liveData_pres.npz'
writeDataLocation = 'liveData_pres.npz'
metadataLocation = 'metadata'
metadata = {'MetaData:' 'Pres '

}

handDataReadVarName = 'dataStore'
targetBoxReadVarName = 'targetBoxLocs'
targetBoxTimeAppearsVarName = 'targetBoxAppearTimes'
targetBoxTimeHitsVarName = 'targetBoxHitTimes'
allBodyDataVarName = 'BodyData'
boxSizeVarName = 'boxSize'
#time to run program
timeProgram = 45 # in seconds


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
