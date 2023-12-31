"""
Stores objects required for the game

"""

# import libraries
import pygame
import os
import numpy as np
from multiprocessing import shared_memory
import time
import sys 


sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')

class gameStatistics():
    """
    Used to store general game information

    """
    def __init__(self,worldx,worldy,LATENCY_TEST,fps,ani,colours,main,timeToReach,
                 FETCHDATAFROMREALTIME,recordData,readData,readLocation,writeDataLocation,
                 metadataLocation,metadata,handDataReadVarName,targetBoxReadVarName,
                 targetBoxTimeAppearsVarName,targetBoxTimeHitsVarName,allBodyDataVarName,
                 boxSizeVarName,timeProgram,reachedBoxStatus,reachedBoxLatch,calibrated,
                 boxHitTimes,enforce,offline,positions,processedRigidBodyParts,
                 leftCornerXBoxLoc,leftCornerYBoxLoc,boxWidth,boxHeight,testMode,readRigidBodies,
                 readAdjustedRigidBodies,showCursorPredictor, cursorMotionDatastoreLocation,runDecoderInLoop):
        self.world = None
        self.calibrationTimeEnd = None
        self.targetStartTime = None
        self.programRunTime = None
        self.steps = None
        self.noTimeStamps = None
        self.worldx = worldx
        self.worldy = worldy
        self.LATENCY_TEST = LATENCY_TEST
        self.fps = fps
        self.ani = ani
        self.colours = colours
        self.main = main
        self.timeToReach = timeToReach
        self.FETCHDATAFROMREALTIME = FETCHDATAFROMREALTIME
        self.recordData = recordData
        self.readData = readData
        self.readLocation = readLocation
        self.writeDataLocation = writeDataLocation
        self.metadataLocation = metadataLocation
        self.metadata = metadata
        self.handDataReadVarName = handDataReadVarName
        self.targetBoxReadVarName = targetBoxReadVarName
        self.targetBoxTimeAppearsVarName = targetBoxTimeAppearsVarName
        self.targetBoxTimeHitsVarName = targetBoxTimeHitsVarName
        self.allBodyDataVarName = allBodyDataVarName
        self.boxSizeVarName = boxSizeVarName
        self.timeProgram = timeProgram
        self.reachedBoxStatus = reachedBoxStatus
        self.reachedBoxLatch = reachedBoxLatch
        self.calibrated = calibrated
        self.boxHitTimes = boxHitTimes
        self.enforce = enforce
        self.offline = offline
        self.positions = positions
        self.processedRigidBodyParts = processedRigidBodyParts
        self.leftCornerXBoxLoc = leftCornerXBoxLoc
        self.leftCornerYBoxLoc = leftCornerYBoxLoc
        self.boxWidth = boxWidth
        self.boxHeight = boxHeight
        self.testMode = testMode
        self.readRigidBodies = readRigidBodies
        self.readAdjustedRigidBodies = readAdjustedRigidBodies
        self.showCursorPredictor = showCursorPredictor
        self.cursorMotionDatastoreLocation = cursorMotionDatastoreLocation
        self.runDecoderInLoop = runDecoderInLoop


class Debugger():
    """
    Used to debug program
    """
    def __init__(self,debugLevel):
        self.debugLevel= debugLevel
        self.test = False

    def disp(self,debugLevel,*var,frequency = None):
        length = len(var)
        if self.debugLevel >= debugLevel:
            if frequency == None or pygame.time.get_ticks() % frequency == 0:
                print('Time:', pygame.time.get_ticks())
                for i in range(length//2):
                    print(var[2*i] , ": ",var[2*i+1])

    def returnDebuggingOutput(self,dataStore,targetBoxLocs, targetBoxHitTimes ,targetBoxAppearTimes,allBodyPartsData,boxSizeVarName,metadata,pointerLocs,gameEngine):
        if self.test:
            return {'Hand Motion':dataStore, 'Target Box Locations': targetBoxLocs,'Target Box Hit times': targetBoxHitTimes ,
                    'Target Box Appear Times' : targetBoxAppearTimes, 'Rigid Body Vectors Datastore': allBodyPartsData,
                    'Box Size': boxSizeVarName, 'Metadata': metadata,' Pointer Location' : pointerLocs, 'GameEngine Metadata': gameEngine}

                
class BoxCollections():
    """
    class to store all boxes
    """
class Box():
    """
    Contains properties for a box
    """
    def __init__(self,leftCornerXBoxLoc,leftCornerYBoxLoc,boxWidth,boxHeight,boxColor,debugger):
        # sets all box properties
        self.leftCornerXBoxLoc = leftCornerXBoxLoc
        self.leftCornerYBoxLoc = leftCornerYBoxLoc
        self.boxWidth = boxWidth
        self.boxHeight = boxHeight
        self.boxColor = boxColor
        self.resetColor = boxColor
        self.dimensions = (leftCornerXBoxLoc, leftCornerYBoxLoc, boxWidth, boxHeight)
        self.readData = False
        self.writeData = False
        self.debugger = debugger
        self.latencyTestActivated = False
        
    
    def prepareForDataWrite(self,boxLocs):
        # tell object program is writing data
        self.writeData = True
        self.writeDatastore = boxLocs
        self.writeDataStoreIteration = 0
        # write the location of the first box and increment iteration
        self.writeDatastore[self.writeDataStoreIteration,0] = self.leftCornerXBoxLoc
        self.writeDatastore[self.writeDataStoreIteration,1] = self.leftCornerYBoxLoc
        self.writeDataStoreIteration += 1

    def prepareForDataRead(self,readLocation,readVarName,player):
        # tell box object that it will be reading data
        self.readData = True
        # feed in location of data object to read
        data = np.load(readLocation)
        self.readDatastore = data[readVarName]
        self.readDataStoreIteration = 0
        
        self.debugger.disp(3,'Old Box left corner x loc', self.leftCornerXBoxLoc, 'Old box left corner y loc', self.leftCornerYBoxLoc)
        # read the location of the first box
        self.leftCornerXBoxLoc = self.readDatastore[self.readDataStoreIteration,0]
        self.leftCornerYBoxLoc = self.readDatastore[self.readDataStoreIteration,1]
        self.readDataStoreIteration += 1
        self.dimensions = (self.leftCornerXBoxLoc, self.leftCornerYBoxLoc, self.boxWidth, self.boxHeight)
        self.debugger.disp(3,'New Box left corner x loc', self.leftCornerXBoxLoc, 'New box left corner y loc', self.leftCornerYBoxLoc)
        # update box position for cursor 
        player.targetBoxXmin = self.leftCornerXBoxLoc
        player.targetBoxXmax = self.leftCornerXBoxLoc + self.boxWidth
        player.targetBoxYmin = self.leftCornerYBoxLoc
        player.targetBoxYmax = self.leftCornerYBoxLoc + self.boxHeight
        return self.leftCornerXBoxLoc, self.leftCornerYBoxLoc 
    


    def resetBoxLocation(self,resetColor = None):
        """
        changes location of box and sets it to reset color
        """
        if resetColor == None:
            self.boxColor = self.resetColor
        else:
            self.boxColor = resetColor
        if self.readData is not True:
            if self.latencyTestActivated is True: 
                self.leftCornerXBoxLoc = 0
                self.leftCornerYBoxLoc = 300
            else:
                self.leftCornerXBoxLoc = np.random.randint(100,1000)
                self.leftCornerYBoxLoc = np.random.randint(100,700)
                if self.writeData is True:
                    # write new locations to datastore
                    self.writeDatastore[self.writeDataStoreIteration,0] = self.leftCornerXBoxLoc
                    self.writeDatastore[self.writeDataStoreIteration,1] = self.leftCornerYBoxLoc
                    self.writeDataStoreIteration += 1


        else:
            
            # read the new location from the datastore
            self.leftCornerXBoxLoc = self.readDatastore[self.readDataStoreIteration,0]
            self.leftCornerYBoxLoc = self.readDatastore[self.readDataStoreIteration,1]
            self.readDataStoreIteration += 1
        #self.debugger.disp(3,'all box dims',self.readDatastore)
       
        self.dimensions = (self.leftCornerXBoxLoc, self.leftCornerYBoxLoc, self.boxWidth, self.boxHeight)
        #self.debugger.disp(3,'actual box dimensions',self.dimensions)
        #self.debugger.disp(3,'theoretical box dimensions',self.readDatastore[self.readDataStoreIteration-1])


class Player(pygame.sprite.Sprite):
    """
    Spawn a cursor
    """

    def __init__(self,targetBox,colours, targetStartTime,worldX,worldY,debugger,cursorPredictor = False,predictorInformation = None):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        # pass debugger
        self.debugger = debugger
        #load the pointer image
        
        self.cursorPredictor = cursorPredictor
        if self.cursorPredictor:
            self.cursorPredictorDatastore = predictorInformation
            self.cursorPredictorDatastoreIteration = 0
            try:
                img = pygame.image.load(os.path.join('Experiment_pointer/images', 'greenDot.png')).convert()
            except FileNotFoundError:
                img = pygame.image.load(os.path.join('images', 'greenDot.png')).convert()
        else:
            try:
                img = pygame.image.load(os.path.join('Experiment_pointer/images', 'dot.png')).convert()
            except FileNotFoundError:
                img = pygame.image.load(os.path.join('images', 'dot.png')).convert()
                
        self.images.append(img)
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.movex = 0 # move along X
        self.movey = 0 # move along Y
        self.latencyTestActivated = False
        self.rightHandIndex = 27
        self.calibrateUsingRightHand = True
        self.userMaxXValue = -10000
        self.userMinXValue = 10000
        self.userMaxYValue = -10000
        self.userMinYValue = 10000
        self.datastore = None
        self.datastoreIteration = 0
        self.calibrated = False
        self.calibrationMatrix = None
        self.calibratedXValue = None
        self.calibratedYValue = None
        self.worldX = worldX
        self.worldY = worldY
        self.xRange = None
        self.yRange = None
        self.rightHandPos = None
        self.rightHandDir = None
        self.readData = False
        self.writeData = False
        self.simulateSharedMemoryOn = False
        self.readAdjustedRigidBodies = False

        # set properties of the target box
        self.boxColor = targetBox.boxColor
        self.targetBoxXmin = targetBox.leftCornerXBoxLoc
        self.targetBoxXmax = targetBox.leftCornerXBoxLoc + targetBox.boxWidth
        self.targetBoxYmin = targetBox.leftCornerYBoxLoc
        self.targetBoxYmax = targetBox.leftCornerYBoxLoc + targetBox.boxHeight

        # set the time at which to spawn the first target
        self.targetStartTime = targetStartTime

        # pass all necessary colours
        self.colours = colours

        # controls live decoding
        self.liveDecoding = False
        
    def setupLiveDecoding(self,gameEngine):
        self.liveDecoding = True
        self.decoderStartTime  = gameEngine.decoderStartTime
        self.modelDecoderType = gameEngine.modelDecoderType
        self.model = np.load(gameEngine.modelReadLocation)
        self.modelCoeff = self.model['modelCoeff']
        if self.modelDecoderType == 'A':
            self.modelIgnoreIdx = 12 # make this more robust after
            self.correctBodyParts = [0,1,2,3,4,5,6,7,8,24,25,26,27,43,44,45,47,48,49] 
        if self.modelDecoderType == 'D':
            self.modelIgnoreIdx = None # make this more robust after
            self.correctBodyParts = [0,1,2,3,4,5,6,7,8,24,25,26,27,43,44,45,47,48,49] 
        # TODO: need to also add ranges which should be tuples corresponding to min and max values to normalise model
        self.DOFmin = self.model['minDOF']
        self.DOFmax = self.model['maxDOF']
        self.DOFOffset = self.model['DOFOffset']

    
    def control(self,x,y):
        """
        set distances to move cursor on next frame from keypad
        """
        if self.readData is not True:
            self.movex = x
            self.movey = -y
    
    def update(self):
        """
        Updates cursor position from keypad
        """
        
        # updates cursor position to be the curr pos plus the next control 
        if self.readData is not True:
            self.rect.x = self.rect.x + self.movex
            self.rect.y = self.rect.y + self.movey
            if self.datastore is not None:
                self.datastore[self.datastoreIteration,0:2] = [self.rect.x,self.rect.y]
                self.datastoreIteration += 1
        else: # read data and increment index
            self.rect.x = self.readDataStore[self.readDataStoreIteration,0] 
            self.rect.y = self.readDataStore[self.readDataStoreIteration,1] 
            self.readDataStoreIteration += 1
        return self.checkIfCursorInBox()

        
    def prepareForDataWrite(self,noTimeStamps):
        noVars = 6
        noBodyParts = 51
        self.noTimeStamps = noTimeStamps
        dataStore = np.zeros((noTimeStamps,noVars)) #
        self.allBodyPartsDatastore = np.zeros((noTimeStamps,noBodyParts,noVars))
        self.allBodyPartsDataStoreIteration = 0
        self.datastore = dataStore # datastore for hand motion only
        # create a datastore for cursor motion
        self.cursorDatastore = np.zeros((noTimeStamps,3)) # 1 for timestamp
        self.cursorDatastoreIndex = 0
        self.targetAppearTimes = np.zeros(100)
        self.targetIndex = 0
        self.writeData = True
        
    
    def checkIfCursorInBox(self):
        """
        checks if cursor inside box, and if so send signal
        """
        # first write cursor loc to datastore
        if self.writeData:
            self.cursorDatastore[self.cursorDatastoreIndex,0] = pygame.time.get_ticks()
            self.cursorDatastore[self.cursorDatastoreIndex,1:3] = [self.rect.x,self.rect.y]
            self.cursorDatastoreIndex += 1
        # check if target has spawned
        if pygame.time.get_ticks() > self.targetStartTime: # if target has spawned
            # check if the cursor is in the target area
            if self.targetBoxXmin <= self.rect.x <= self.targetBoxXmax and self.targetBoxYmin <= self.rect.y <= self.targetBoxYmax:
                return 1
            else:
                return 0
        else:
            return 0
    
    def prepareForDataRead(self,readDataLocation,readDataVarName,allBodyDataVarName):
        data = np.load(readDataLocation)
        bodyData = data[allBodyDataVarName]
        data = data[readDataVarName]
        if self.simulateSharedMemoryOn:
            self.bodyDataStore = bodyData
            self.bodyDataStoreIteration = 0
        self.readDataStore = data
        self.readDataStoreIteration = 0
        self.readData = True

    def initSharedMemory(self,BODY_PART_MEM,noDataTypes,noBodyParts):
        if self.readData is not True:
            self.debugger.disp(3,'Shared memory is being used', '')
            self.sharedMemName = BODY_PART_MEM
            self.sharedMemShape = (noBodyParts,noDataTypes)
            self.sharedMemSize =  noDataTypes * noBodyParts
        else:
            self.debugger.disp(3,'Shared memory is not being used', '')
            if self.simulateSharedMemoryOn:
                self.sharedMemName = BODY_PART_MEM
                self.sharedMemShape = (noBodyParts,noDataTypes)
                self.sharedMemSize =  noDataTypes * noBodyParts
                shared_block = shared_memory.SharedMemory(size= self.sharedMemSize * 8, name=self.sharedMemName, create=True)
                shared_array = np.ndarray(shape=self.sharedMemShape, dtype=np.float64, buffer=shared_block.buf)
    
    def processData(self):
        self.allBodyPartsDatastore = np.reshape(self.allBodyPartsDatastore,(self.noTimeStamps,-1))

    def fetchSharedMemoryData(self):
        if not self.cursorPredictor:
            if self.readData is not True:
                shared_block = shared_memory.SharedMemory(size= self.sharedMemSize * 8, name=self.sharedMemName, create=False)
                shared_array = np.ndarray(shape=self.sharedMemShape, dtype=np.float64, buffer=shared_block.buf)
                rightHandData = np.array(shared_array[ self.rightHandIndex])
                if self.datastore is not None: # record data if requested
                    self.datastore[self.datastoreIteration,:] = rightHandData[:6]
                    self.datastoreIteration += 1
                    # now write all body part info to database
                    self.allBodyPartsDatastore[self.allBodyPartsDataStoreIteration,:,:] =  np.array(shared_array[:,0:6])
                    self.allBodyPartsDataStoreIteration += 1
                # self.rightHandPos = rightHandData[0:3]
                # self.rightHandDir = rightHandData[3:6]
            else:
                # read from datastore and increment index
                rightHandData = self.readDataStore[self.readDataStoreIteration] 
                self.readDataStoreIteration += 1
                


            # both workflows have this adjustment
            self.rightHandPos = np.matmul(self.calibrationMatrix,rightHandData[0:3])
            self.rightHandDir = np.matmul(self.calibrationMatrix,rightHandData[3:6])
            
            if self.readData is True and self.simulateSharedMemoryOn:
                shared_block = shared_memory.SharedMemory(size= self.sharedMemSize * 8, name=self.sharedMemName, create=False)
                shared_array = np.ndarray(shape=self.sharedMemShape, dtype=np.float64, buffer=shared_block.buf)
                if self.readAdjustedRigidBodies:
                    Q = np.zeros((6,6))
                    Q[0:3,0:3] = self.calibrationMatrix
                    Q[3:6,3:6] = self.calibrationMatrix 
                    shared_array[:,:6] = np.matmul(Q,self.bodyDataStore[self.bodyDataStoreIteration].reshape(51,6).transpose() ).transpose()
                else:
                    shared_array[:,:6] = self.bodyDataStore[self.bodyDataStoreIteration].reshape(51,6) 
                self.bodyDataStoreIteration += 1

            if self.calibrated is not True:
                # as game x is typically the body y plane (right) and game y is the body z plane (up) 
                self.userMaxXValue = max(self.rightHandPos[1],self.userMaxXValue)
                self.userMaxYValue = max(self.rightHandPos[2],self.userMaxYValue)
                self.userMinXValue = min(self.rightHandPos[1],self.userMinXValue)
                self.userMinYValue = min(self.rightHandPos[2],self.userMinYValue)
            
            if self.liveDecoding and pygame.time.get_ticks() > self.decoderStartTime:
                if not hasattr(self, 'fullCalibrationMatrix'):
                    self.fullCalibrationMatrix = np.zeros((6,6))
                    self.fullCalibrationMatrix[0:3,0:3] = self.calibrationMatrix
                    self.fullCalibrationMatrix[3:6,3:6] = self.calibrationMatrix 
                if False:
                    tmpRigBodyArray = shared_array[self.correctBodyParts,:6]
                    tmpRigBodyArray = np.matmul(self.fullCalibrationMatrix,tmpRigBodyArray.transpose()).transpose().reshape(-1,1)
                    tmpRigBodyArray = np.array([(tmpRigBodyArray[i] - self.DOFmin[i]) / (self.DOFmax[i] - self.DOFmin[i] + self.DOFOffset) for i in range(0,len(self.DOFmin))])
                    idxRightHand = self.modelIgnoreIdx * 6
                    tmpArray = np.zeros(108)
                    tmpArray[0:idxRightHand] = tmpRigBodyArray[0:idxRightHand,0]
                    tmpArray[idxRightHand:] = tmpRigBodyArray[idxRightHand+6:,0]
                    self.xposDECODE = np.matmul(self.modelCoeff[0],tmpArray.transpose())
                    self.yposDECODE = np.matmul(self.modelCoeff[1],tmpArray.transpose())
                
                if True:
                    tmpRigBodyArray = shared_array[self.correctBodyParts,:6] # idx 12
                    tmpRigBodyArray = np.matmul(self.fullCalibrationMatrix,tmpRigBodyArray.transpose()).transpose().reshape(-1,1)
                    tmpRigBodyArray = np.array([(tmpRigBodyArray[i] - self.DOFmin[i]) / (self.DOFmax[i] - self.DOFmin[i] + self.DOFOffset) for i in range(0,len(self.DOFmin))])
                    idxRightHand = self.modelIgnoreIdx * 6
                    tmpArray = tmpRigBodyArray[idxRightHand:idxRightHand+6] 
                    self.yposDECODE = -np.matmul(self.modelCoeff[0],tmpArray)
                    self.xposDECODE = -np.matmul(self.modelCoeff[1],tmpArray)
                # ignore relevant indexes
                
                # control motion using model
                # for now decoderStartTime should be higher than calibration time
                #TODO: first multiply rigid body vector by calibration matrix
                #TODO: resize each rigid body by ranges # self.ranges[i] = (min,max) for ith rigid body dim
                #TODO: convert rigid bodies to cursor pos
                #self.cursorPosLive = ...


    def finishCalibrationStage(self):
        print('Calibration stage is now over')
        # add code to make calibration even with respect to far distance to middle
        self.xRange = self.userMaxXValue - self.userMinXValue
        self.yRange = self.userMaxYValue - self.userMinYValue
        self.calibrated = True
        self.targetStartTime = pygame.time.get_ticks() + 3000
        # add target appear time to array of target appear times
        if self.datastore is not None:
            self.targetAppearTimes[self.targetIndex] = self.targetStartTime
            self.targetIndex += 1
        return self.targetStartTime


    def calcCursorPosFromHandData(self):
        #print(self.xRange,self.yRange)
        if self.liveDecoding and pygame.time.get_ticks() > self.decoderStartTime:
            self.rect.x = self.xposDECODE * self.worldX
            self.rect.y = self.yposDECODE * self.worldY
            self.debugger.disp(3,'X pos', self.rect.x,frequency = 50)
            self.debugger.disp(3,'Y pos', self.rect.y,frequency = 50)
            return self.checkIfCursorInBox()

        elif not self.cursorPredictor:
            normalised_x_val = 1 -  (self.rightHandPos[1] - self.userMinXValue) / self.xRange
            normalised_y_Val = 1 - (self.rightHandPos[2] - self.userMinYValue) / self.yRange
            #print(normalised_x_val)
            #print(normalised_y_Val)
            self.rect.x = normalised_x_val * self.worldX
            self.rect.y = normalised_y_Val * self.worldY
            return self.checkIfCursorInBox()
        else:
            self.rect.x  = self.cursorPredictorDatastore[self.cursorPredictorDatastoreIteration,0] * self.worldX
            self.rect.y  = self.cursorPredictorDatastore[self.cursorPredictorDatastoreIteration,1] * self.worldY
            self.cursorPredictorDatastoreIteration += 1
        

    def enterCalibrationStage(self):

        print('Before we start, the calibration stage must be undertaken')
        print('Face upright and standing towards the computer and point the right hand forward')
        print('Calibration will start in 5')
        time.sleep(1)
        print('4')
        time.sleep(1)
        print('3')
        time.sleep(1)
        print('2')
        time.sleep(1)
        print('1')
        time.sleep(1)
        print('Calibrating - point your hand and arm forwards and keep it straight')

        if self.calibrateUsingRightHand:
            if self.readData is not True:
                shared_block = shared_memory.SharedMemory(size= self.sharedMemSize * 8, name=self.sharedMemName, create=False)
                shared_array = np.ndarray(shape=self.sharedMemShape, dtype=np.float64, buffer=shared_block.buf)
                rightHandData = np.array(shared_array[27,:]) # idx of right hand

                self.datastore[self.datastoreIteration,:] = rightHandData[:6]
                self.datastoreIteration += 1
                # now write all body part info to database
                self.allBodyPartsDatastore[self.allBodyPartsDataStoreIteration,:,:] =  np.array(shared_array[:,0:6])
                self.allBodyPartsDataStoreIteration += 1
            else:
                # read data from datastore and increment index
                rightHandData = self.readDataStore[self.readDataStoreIteration] 
                self.readDataStoreIteration += 1

        else:
            raise('Need to code this in')
        calibrationFromVector = rightHandData[3:6]
        
        calibrationToVector = np.array([1,0,0])
        self.calcCalibrationConstants(calibrationToVector,calibrationFromVector)
        self.calibratedXValue, self.calibratedYValue =  np.matmul(self.calibrationMatrix,np.array(rightHandData[0:3]))[1:3]

        print('Correct plane has been calibrated')
        print('The program will now calibrate the x and y range')
        print('Please move your right arm wherever possible')
        print('The program will form this new range based on your motion in the next 7 seconds')
    

        

    def calcCalibrationConstants(self,calibrationToVector, calibrationFromVector):
        """
        attempts to calibrate for person standing off x axis by finding the transformation
        matrix to transform off axis motion to the standard axes
        returns a transformation matrix that can convert directions and positions 
        """
        # calculate thetha from dot product
        thetha_rad = np.arccos(np.dot(calibrationToVector,calibrationFromVector)/(np.linalg.norm(calibrationToVector) * np.linalg.norm(calibrationFromVector)))

        # calculate Q
        Q = np.zeros((3,3))
        Q[0,0] = np.cos(thetha_rad)
        Q[1,1] = Q[0,0]
        Q[0,1] = np.sin(thetha_rad)
        Q[1,0] = - Q[0,1]
        Q[2,2] = 1

        self.calibrationMatrix = Q.transpose()


    
    def updatepos(self,x,y,enforce,offline,positions):
        # note will need to enforce 0<x<960 and 0 < y < 720
        # this is activated if data is streamed
        if enforce and offline: 
            y = 720 * (y/2000)
            x = (x + 600)
        elif enforce and not offline and not positions: # i.e. for position vectors
            y = (y+1)/1 * 360 # 
            x = (x+0.25) * 1200
        elif enforce and not offline and positions:
            y = (y+1)/1 * 600 # 
            x = (x-1)/4 * 800   # 1600 - 2400
        self.rect.x = x
        self.rect.y = y
        return self.checkIfCursorInBox()

    def reset(self,targetBox):
        # changes the target box to be the new dimensions, mainly to assess whether the cursor is in the target box
        self.targetBoxXmin = targetBox.leftCornerXBoxLoc
        self.targetBoxXmax = targetBox.leftCornerXBoxLoc + targetBox.boxWidth
        self.targetBoxYmin = targetBox.leftCornerYBoxLoc
        self.targetBoxYmax = targetBox.leftCornerYBoxLoc + targetBox.boxHeight
        self.boxColor = targetBox.boxColor
        # add time of target appearing so it can be seen
        if self.datastore is not None:
            self.targetAppearTimes[self.targetIndex] = pygame.time.get_ticks()
            self.targetIndex += 1
    

            
