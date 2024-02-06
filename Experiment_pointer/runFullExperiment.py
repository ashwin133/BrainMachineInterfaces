"""
This file contains workflows for the experiment (simple game)



"""

import pygame
import sys
import os
from multiprocessing import shared_memory
import numpy as np
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')


# import variables and objects used and run setup
from Experiment_pointer.variables import gameEngine
from Experiment_pointer.objects import *
from Experiment_pointer.setup import runSetup, endProgram
from Experiment_pointer.runner import runGame
from Experiment_pointer.experimentFunctions import *


# set world x and world y and fps for whole experiment
worldx = 1100
worldy = 800
fps = 60

# Set this True if data has already been gathered from the experiment 
# and only data analysis needs to be performed
runPostExperimentalAnalysisOnly = True

saveGameLocation = "Experiment_pointer/PointerExperimentData/Ash_27_01_15_39" # must have format of "Name_dd_mm__hh_mm_metadata" with no file extension


if runPostExperimentalAnalysisOnly == False:

#     # --- Phase 1: 4 training trials, each 3 minutes ---
    # ---- TRAINING TRIAL 1
    print("--- RUNNING TRAINING TRIAL 1 --- DURATION: 2 MINUTES --- \n")

    # Set Game 1 save Locations
    trainingGameSaveLocation1 = saveGameLocation + "_training1.npz"
    trainingGameSaveLocationPkl1 = saveGameLocation + "_training1.pkl"

    initiateUserConfirmToProceed()
    gameEngine_trial1 = gameEngine
    gameEngine_trial1 = configureForTrainingSetup(gameEngine=gameEngine_trial1,worldx=worldx,worldy=worldy,fps=fps,saveLocation=trainingGameSaveLocation1,saveLocationPkl=trainingGameSaveLocationPkl1)
    player,targetBox,gameEngine_trial1, clock, player_list,debugger,cursorPredictor =runSetup(gameEngine=gameEngine_trial1)
    runGame(gameEngine=gameEngine_trial1, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list,clock = clock)

    # check if data saved an inform user if game has not saved
    verifySaved1 = verifyGameSaveData(trainingGameSaveLocation1,trainingGameSaveLocationPkl1)
    informUserGameSave(verifySaved=verifySaved1 )

    # Delete data after saved
    del player,targetBox,gameEngine_trial1, clock, player_list,debugger,cursorPredictor 

    initiateUserConfirmToProceed(forBreak=True)

    # ---- TRAINING TRIAL 2
    print("--- RUNNING TRAINING TRIAL 2 --- DURATION: 2 MINUTES --- \n")

    # Set Game 2 save Locations
    trainingGameSaveLocation2 = saveGameLocation + "_training2.npz"
    trainingGameSaveLocationPkl2 = saveGameLocation + "_training2.pkl"

    initiateUserConfirmToProceed()
    gameEngine_trial2 = gameEngine
    gameEngine_trial2 = configureForTrainingSetup(gameEngine=gameEngine_trial2,worldx=worldx,worldy=worldy,fps=fps,saveLocation=trainingGameSaveLocation2,saveLocationPkl=trainingGameSaveLocationPkl2)
    player,targetBox,gameEngine_trial2, clock, player_list,debugger,cursorPredictor =runSetup(gameEngine=gameEngine_trial2)
    runGame(gameEngine=gameEngine_trial2, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list,clock = clock)

    # check if data saved an inform user if game has not saved
    verifySaved2 = verifyGameSaveData(trainingGameSaveLocation2,trainingGameSaveLocationPkl2)
    informUserGameSave(verifySaved=verifySaved2 )

    # Delete data after saved
    del player,targetBox,gameEngine_trial2, clock, player_list,debugger,cursorPredictor 

    initiateUserConfirmToProceed(forBreak=True)



    # ---- TRAINING TRIAL 3
    print("--- RUNNING TRAINING TRIAL 3 --- DURATION: 2 MINUTES --- \n")

    # Set Game 3 save Locations
    trainingGameSaveLocation3 = saveGameLocation + "_training3.npz"
    trainingGameSaveLocationPkl3 = saveGameLocation + "_training3.pkl"

    initiateUserConfirmToProceed()
    gameEngine_trial3 = gameEngine
    gameEngine_trial3 = configureForTrainingSetup(gameEngine=gameEngine_trial3,worldx=worldx,worldy=worldy,fps=fps,saveLocation=trainingGameSaveLocation3,saveLocationPkl=trainingGameSaveLocationPkl3)
    player,targetBox,gameEngine_trial3, clock, player_list,debugger,cursorPredictor =runSetup(gameEngine=gameEngine_trial3)
    runGame(gameEngine=gameEngine_trial3, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list,clock = clock)

    # check if data saved an inform user if game has not saved
    verifySaved3 = verifyGameSaveData(trainingGameSaveLocation3,trainingGameSaveLocationPkl3)
    informUserGameSave(verifySaved=verifySaved3 )

    # Delete data after saved
    del player,targetBox,gameEngine_trial3, clock, player_list,debugger,cursorPredictor 

    initiateUserConfirmToProceed(forBreak=True)


    # ---- TRAINING TRIAL 4
    print("--- RUNNING TRAINING TRIAL 4 --- DURATION: 2 MINUTES --- \n")

    # Set Game 4 save Locations
    trainingGameSaveLocation4 = saveGameLocation + "_training4.npz"
    trainingGameSaveLocationPkl4 = saveGameLocation + "_training4.pkl"

    initiateUserConfirmToProceed()
    gameEngine_trial4 = gameEngine
    gameEngine_trial4 = configureForTrainingSetup(gameEngine=gameEngine_trial4,worldx=worldx,worldy=worldy,fps=fps,saveLocation=trainingGameSaveLocation4,saveLocationPkl=trainingGameSaveLocationPkl4)
    player,targetBox,gameEngine_trial4, clock, player_list,debugger,cursorPredictor =runSetup(gameEngine=gameEngine_trial4)
    runGame(gameEngine=gameEngine_trial4, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list,clock = clock)

    # check if data saved an inform user if game has not saved
    verifySaved4 = verifyGameSaveData(trainingGameSaveLocation4,trainingGameSaveLocationPkl4)
    informUserGameSave(verifySaved=verifySaved4 )

    # Delete data after saved
    del player,targetBox,gameEngine_trial4, clock, player_list,debugger,cursorPredictor 

    initiateUserConfirmToProceed(forBreak=True)


    # ---- TEST TRIAL 
    print("--- RUNNING TEST TRIAL --- DURATION: 2 MINUTES --- \n")

    # Set Test game save Locations
    testGameSaveLocation = saveGameLocation + "_test.npz"
    testGameSaveLocationPkl = saveGameLocation + "_test.pkl"

    initiateUserConfirmToProceed()
    gameEngine_testTrial = gameEngine
    gameEngine_testTrial = configureForTrainingSetup(gameEngine=gameEngine_testTrial,worldx=worldx,worldy=worldy,fps=fps,saveLocation=testGameSaveLocation,saveLocationPkl=testGameSaveLocationPkl)
    player,targetBox,gameEngine_testTrial, clock, player_list,debugger,cursorPredictor =runSetup(gameEngine=gameEngine_testTrial)
    runGame(gameEngine=gameEngine_testTrial, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list,clock = clock)

    # check if data saved an inform user if game has not saved
    verifySaved_test = verifyGameSaveData(testGameSaveLocation,testGameSaveLocationPkl)
    informUserGameSave(verifySaved=verifySaved_test )

    # Delete data after saved
    del player,targetBox,gameEngine_testTrial, clock, player_list,debugger,cursorPredictor 

    initiateUserConfirmToProceed(forBreak=True)

    print("--- The initial experimental phase has now been finished ---")

    """
    --- END OF PHASE 1 ----
    """

    # 
    # 
"""
Phase 2 - train decoders
"""

# A : Linear : 0.01: no ignore
angularInfoDictA = fitModelToData(mode = 'RigidBodiesSetA',tester = 'linear', \
compPca = None,savePath=saveGameLocation, colorMap=colorMap,plot=False,DOFOffset= 0.01,ignoreTargetMotionTimesLessThan=0)

# Save model 
np.savez(saveGameLocation + '_linearRigidBodyAModel.npz', modelCoeff = angularInfoDictA['Coeff'],modelIntercept = angularInfoDictA['Intercept'],minDOF = angularInfoDictA['MinDOF'],
        maxDOF = angularInfoDictA['MaxDOF'], DOFOffset = angularInfoDictA['DOFOffset'], predCursorPos = angularInfoDictA['PredCursorPos'])

# # B : Linear : 0.01 :ignore > 600
angularInfoDictB = fitModelToData(mode = 'RigidBodiesSetB',tester = 'linear', \
compPca = None,savePath=saveGameLocation, colorMap=colorMap,plot=False,DOFOffset= 0.01,ignoreTargetMotionTimesLessThan=600)

# Save model 
np.savez(saveGameLocation + '_linearRigidBodyBModel.npz', modelCoeff = angularInfoDictB['Coeff'],modelIntercept = angularInfoDictB['Intercept'],minDOF = angularInfoDictB['MinDOF'],
        maxDOF = angularInfoDictB['MaxDOF'], DOFOffset = angularInfoDictB['DOFOffset'], predCursorPos = angularInfoDictB['PredCursorPos'])


# # C : Linear : 0.05 :ignore > 0
angularInfoDictC = fitModelToData(mode = 'RigidBodiesSetC',tester = 'linear', \
compPca = None, savePath=saveGameLocation,colorMap=colorMap,plot=False,DOFOffset= 0.05,ignoreTargetMotionTimesLessThan=0)

# Save model 
np.savez( saveGameLocation + '_linearRigidBodyCModel.npz', modelCoeff = angularInfoDictC['Coeff'],modelIntercept = angularInfoDictC['Intercept'],minDOF = angularInfoDictC['MinDOF'],
        maxDOF = angularInfoDictC['MaxDOF'], DOFOffset = angularInfoDictC['DOFOffset'], predCursorPos = angularInfoDictC['PredCursorPos'])


# # D : Linear : 0.1 :ignore > 0
angularInfoDictD = fitModelToData(mode = 'RigidBodiesSetD',tester = 'linear', \
compPca = None, savePath=saveGameLocation,colorMap=colorMap,plot=False,DOFOffset= 0.1,ignoreTargetMotionTimesLessThan=0)

# Save model 
np.savez(saveGameLocation + '_linearRigidBodyDModel.npz', modelCoeff = angularInfoDictD['Coeff'],modelIntercept = angularInfoDictD['Intercept'],minDOF = angularInfoDictD['MinDOF'],
        maxDOF = angularInfoDictD['MaxDOF'], DOFOffset = angularInfoDictD['DOFOffset'], predCursorPos = angularInfoDictD['PredCursorPos'])


# # E : Linear : 0.1 :ignore > 0, angles only
angularInfoDictE = fitModelToData(mode = 'RigidBodiesSetE',tester = 'linear', \
compPca = None, savePath=saveGameLocation,colorMap=colorMap,plot=False,DOFOffset= 0.1,ignoreTargetMotionTimesLessThan=0)

np.savez(saveGameLocation + '_linearRigidBodyEModel.npz', modelCoeff = angularInfoDictE['Coeff'],modelIntercept = angularInfoDictE['Intercept'],minDOF = angularInfoDictE['MinDOF'],
        maxDOF = angularInfoDictE['MaxDOF'], DOFOffset = angularInfoDictE['DOFOffset'], predCursorPos = angularInfoDictE['PredCursorPos'])


# # F : Linear : 0.1 :ignore > 0, angles only
angularInfoDictF = fitModelToData(mode = 'RigidBodiesSetF',tester = 'linear', \
compPca = None, savePath=saveGameLocation,colorMap=colorMap,plot=False,DOFOffset= 0.1,ignoreTargetMotionTimesLessThan=0)

np.savez(saveGameLocation + '_linearRigidBodyFModel.npz', modelCoeff = angularInfoDictF['Coeff'],modelIntercept = angularInfoDictF['Intercept'],minDOF = angularInfoDictF['MinDOF'],
        maxDOF = angularInfoDictF['MaxDOF'], DOFOffset = angularInfoDictF['DOFOffset'], predCursorPos = angularInfoDictF['PredCursorPos'])


# # G : Linear : 0.1 :ignore > 0, angles only
angularInfoDictG = fitModelToData(mode = 'RigidBodiesSetG',tester = 'linear', \
compPca = None, savePath=saveGameLocation,colorMap=colorMap,plot=False,DOFOffset= 0.1,ignoreTargetMotionTimesLessThan=0)

np.savez(saveGameLocation + '_linearRigidBodyGModel.npz', modelCoeff = angularInfoDictG['Coeff'],modelIntercept = angularInfoDictG['Intercept'],minDOF = angularInfoDictG['MinDOF'],
        maxDOF = angularInfoDictG['MaxDOF'], DOFOffset = angularInfoDictG['DOFOffset'], predCursorPos = angularInfoDictG['PredCursorPos'])

# # H : Linear : 0.1 :ignore > 0, angles only
angularInfoDictH = fitModelToData(mode = 'RigidBodiesSetH',tester = 'linear', \
compPca = None, savePath=saveGameLocation,colorMap=colorMap,plot=False,DOFOffset= 0.1,ignoreTargetMotionTimesLessThan=0)

np.savez(saveGameLocation + '_linearRigidBodyHModel.npz', modelCoeff = angularInfoDictH['Coeff'],modelIntercept = angularInfoDictH['Intercept'],minDOF = angularInfoDictH['MinDOF'],
        maxDOF = angularInfoDictH['MaxDOF'], DOFOffset = angularInfoDictH['DOFOffset'], predCursorPos = angularInfoDictH['PredCursorPos'])

# # I : Linear : 0.1 :ignore > 0, angles only
angularInfoDictI = fitModelToData(mode = 'RigidBodiesSetI',tester = 'linear', \
compPca = None, savePath=saveGameLocation,colorMap=colorMap,plot=False,DOFOffset= 0.1,ignoreTargetMotionTimesLessThan=0)

np.savez(saveGameLocation + '_linearRigidBodyIModel.npz', modelCoeff = angularInfoDictI['Coeff'],modelIntercept = angularInfoDictI['Intercept'],minDOF = angularInfoDictI['MinDOF'],
        maxDOF = angularInfoDictI['MaxDOF'], DOFOffset = angularInfoDictI['DOFOffset'], predCursorPos = angularInfoDictI['PredCursorPos'])

# # J : Linear : 0.1 :ignore > 0, angles only
angularInfoDictJ = fitModelToData(mode = 'RigidBodiesSetJ',tester = 'linear', \
compPca = None, savePath=saveGameLocation,colorMap=colorMap,plot=False,DOFOffset= 0.1,ignoreTargetMotionTimesLessThan=0)

np.savez(saveGameLocation + '_linearRigidBodyJModel.npz', modelCoeff = angularInfoDictJ['Coeff'],modelIntercept = angularInfoDictJ['Intercept'],minDOF = angularInfoDictJ['MinDOF'],
        maxDOF = angularInfoDictJ['MaxDOF'], DOFOffset = angularInfoDictJ['DOFOffset'], predCursorPos = angularInfoDictJ['PredCursorPos'])







"""
END OF PHASE 2

"""

"""
PHASE 3: Testing decoders in the closed loop
# """
print("--- PHASE 3: Testing out decoders in the closed loop ---")

# # --- Decoder E ---
# print("--- RUNNING DECODER TRIAL E --- DURATION: 3 MINUTES --- \n")

# # Set Test game save Locations
# decoderESaveLocation = saveGameLocation + "_usingDecoderE.npz"
# decoderESaveLocationPkl = saveGameLocation + "_usingDecoderE.pkl"
# decoderEMdlLocation = saveGameLocation + '_linearRigidBodyEModel.npz'

# initiateUserConfirmToProceed()
# gameEngine_decoderE = gameEngine
# gameEngine_decoderE = configureForDecoder(gameEngine=gameEngine_decoderE,worldx=worldx,worldy=worldy,fps=fps,saveLocation=decoderESaveLocation,
#                 saveLocationPkl=decoderESaveLocationPkl,decoder= 'E', decoderLocation = decoderEMdlLocation)

# player,targetBox,gameEngine_decoderE, clock, player_list,debugger,cursorPredictor =runSetup(gameEngine=gameEngine_decoderE)
# runGame(gameEngine=gameEngine_decoderE, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list,clock = clock)

# # check if data saved an inform user if game has not saved
# verifySaved_decoderE = verifyGameSaveData(decoderESaveLocation,decoderESaveLocationPkl)
# informUserGameSave(verifySaved=verifySaved_decoderE )

# # Delete data after saved
# del player,targetBox,gameEngine_decoderE, clock, player_list,debugger,cursorPredictor 

# initiateUserConfirmToProceed(forBreak=True)


# --- Decoder F ---
print("--- RUNNING DECODER TRIAL F --- DURATION: 3 MINUTES --- \n")

# Set Test game save Locations
decoderFSaveLocation = saveGameLocation + "_usingDecoderF.npz"
decoderFSaveLocationPkl = saveGameLocation + "_usingDecoderF.pkl"
decoderFMdlLocation = saveGameLocation + '_linearRigidBodyFModel.npz'

initiateUserConfirmToProceed()
gameEngine_decoderF = gameEngine
gameEngine_decoderF = configureForDecoder(gameEngine=gameEngine_decoderF,worldx=worldx,worldy=worldy,fps=fps,saveLocation=decoderFSaveLocation,
                saveLocationPkl=decoderFSaveLocationPkl,decoder= 'F', decoderLocation = decoderFMdlLocation)

player,targetBox,gameEngine_decoderF, clock, player_list,debugger,cursorPredictor =runSetup(gameEngine=gameEngine_decoderF)
runGame(gameEngine=gameEngine_decoderF, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list,clock = clock)

# check if data saved an inform user if game has not saved
verifySaved_decoderF = verifyGameSaveData(decoderFSaveLocation,decoderFSaveLocationPkl)
informUserGameSave(verifySaved=verifySaved_decoderF )

# Delete data after saved
del player,targetBox,gameEngine_decoderF, clock, player_list,debugger,cursorPredictor 

initiateUserConfirmToProceed(forBreak=True)


# --- Decoder G ---
print("--- RUNNING DECODER TRIAL G --- DURATION: 3 MINUTES --- \n")

# Set Test game save Locations
decoderGSaveLocation = saveGameLocation + "_usingDecoderG.npz"
decoderGSaveLocationPkl = saveGameLocation + "_usingDecoderG.pkl"
decoderGMdlLocation = saveGameLocation + '_linearRigidBodyGModel.npz'

initiateUserConfirmToProceed()
gameEngine_decoderG = gameEngine
gameEngine_decoderG = configureForDecoder(gameEngine=gameEngine_decoderG,worldx=worldx,worldy=worldy,fps=fps,saveLocation=decoderGSaveLocation,
                saveLocationPkl=decoderGSaveLocationPkl,decoder= 'G', decoderLocation = decoderGMdlLocation)

player,targetBox,gameEngine_decoderG, clock, player_list,debugger,cursorPredictor =runSetup(gameEngine=gameEngine_decoderG)
runGame(gameEngine=gameEngine_decoderG, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list,clock = clock)

# check if data saved an inform user if game has not saved
verifySaved_decoderG = verifyGameSaveData(decoderGSaveLocation,decoderGSaveLocationPkl)
informUserGameSave(verifySaved=verifySaved_decoderG )

# Delete data after saved
del player,targetBox,gameEngine_decoderG, clock, player_list,debugger,cursorPredictor 

initiateUserConfirmToProceed(forBreak=True)


# --- Decoder H ---
print("--- RUNNING DECODER TRIAL H --- DURATION: 3 MINUTES --- \n")

# Set Test game save Locations
decoderHSaveLocation = saveGameLocation + "_usingDecoderH.npz"
decoderHSaveLocationPkl = saveGameLocation + "_usingDecoderH.pkl"
decoderHMdlLocation = saveGameLocation + '_linearRigidBodyHModel.npz'

initiateUserConfirmToProceed()
gameEngine_decoderH = gameEngine
gameEngine_decoderH = configureForDecoder(gameEngine=gameEngine_decoderH,worldx=worldx,worldy=worldy,fps=fps,saveLocation=decoderHSaveLocation,
                saveLocationPkl=decoderHSaveLocationPkl,decoder= 'H', decoderLocation = decoderHMdlLocation)

player,targetBox,gameEngine_decoderH, clock, player_list,debugger,cursorPredictor =runSetup(gameEngine=gameEngine_decoderH)
runGame(gameEngine=gameEngine_decoderH, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list,clock = clock)

# check if data saved an inform user if game has not saved
verifySaved_decoderH = verifyGameSaveData(decoderHSaveLocation,decoderHSaveLocationPkl)
informUserGameSave(verifySaved=verifySaved_decoderH )

# Delete data after saved
del player,targetBox,gameEngine_decoderH, clock, player_list,debugger,cursorPredictor 

initiateUserConfirmToProceed(forBreak=True)

# --- Decoder I ---
print("--- RUNNING DECODER TRIAL I --- DURATION: 3 MINUTES --- \n")

# Set Test game save Locations
decoderISaveLocation = saveGameLocation + "_usingDecoderI.npz"
decoderISaveLocationPkl = saveGameLocation + "_usingDecoderI.pkl"
decoderIMdlLocation = saveGameLocation + '_linearRigidBodyIModel.npz'

initiateUserConfirmToProceed()
gameEngine_decoderI = gameEngine
gameEngine_decoderI = configureForDecoder(gameEngine=gameEngine_decoderI,worldx=worldx,worldy=worldy,fps=fps,saveLocation=decoderISaveLocation,
                saveLocationPkl=decoderISaveLocationPkl,decoder= 'I', decoderLocation = decoderIMdlLocation)

player,targetBox,gameEngine_decoderI, clock, player_list,debugger,cursorPredictor =runSetup(gameEngine=gameEngine_decoderI)
runGame(gameEngine=gameEngine_decoderI, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list,clock = clock)

# check if data saved an inform user if game has not saved
verifySaved_decoderI = verifyGameSaveData(decoderISaveLocation,decoderISaveLocationPkl)
informUserGameSave(verifySaved=verifySaved_decoderI )

# Delete data after saved
del player,targetBox,gameEngine_decoderI, clock, player_list,debugger,cursorPredictor 

initiateUserConfirmToProceed(forBreak=True)

# --- Decoder J ---
print("--- RUNNING DECODER TRIAL J --- DURATION: 3 MINUTES --- \n")

# Set Test game save Locations
decoderJSaveLocation = saveGameLocation + "_usingDecoderJ.npz"
decoderJSaveLocationPkl = saveGameLocation + "_usingDecoderJ.pkl"
decoderJMdlLocation = saveGameLocation + '_linearRigidBodyJModel.npz'

initiateUserConfirmToProceed()
gameEngine_decoderJ = gameEngine
gameEngine_decoderJ = configureForDecoder(gameEngine=gameEngine_decoderJ,worldx=worldx,worldy=worldy,fps=fps,saveLocation=decoderJSaveLocation,
                saveLocationPkl=decoderJSaveLocationPkl,decoder= 'J', decoderLocation = decoderJMdlLocation)

player,targetBox,gameEngine_decoderJ, clock, player_list,debugger,cursorPredictor =runSetup(gameEngine=gameEngine_decoderJ)
runGame(gameEngine=gameEngine_decoderJ, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list,clock = clock)

# check if data saved an inform user if game has not saved
verifySaved_decoderJ = verifyGameSaveData(decoderJSaveLocation,decoderJSaveLocationPkl)
informUserGameSave(verifySaved=verifySaved_decoderJ )

# Delete data after saved
del player,targetBox,gameEngine_decoderJ, clock, player_list,debugger,cursorPredictor 

initiateUserConfirmToProceed(forBreak=True)

# # --- Decoder A ---
print("--- RUNNING DECODER TRIAL A --- DURATION: 3 MINUTES --- \n")

# Set Test game save Locations
decoderASaveLocation = saveGameLocation + "_usingDecoderA.npz"
decoderASaveLocationPkl = saveGameLocation + "_usingDecoderA.pkl"
decoderAMdlLocation = saveGameLocation + '_linearRigidBodyAModel.npz'

initiateUserConfirmToProceed()
gameEngine_decoderA = gameEngine
gameEngine_decoderA = configureForDecoder(gameEngine=gameEngine_decoderA,worldx=worldx,worldy=worldy,fps=fps,saveLocation=decoderASaveLocation,
                saveLocationPkl=decoderASaveLocationPkl,decoder= 'A', decoderLocation = decoderAMdlLocation)

player,targetBox,gameEngine_decoderA, clock, player_list,debugger,cursorPredictor =runSetup(gameEngine=gameEngine_decoderA)
runGame(gameEngine=gameEngine_decoderA, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list,clock = clock)

# check if data saved an inform user if game has not saved
verifySaved_decoderA = verifyGameSaveData(decoderASaveLocation,decoderASaveLocationPkl)
informUserGameSave(verifySaved=verifySaved_decoderA )

# Delete data after saved
del player,targetBox,gameEngine_decoderA, clock, player_list,debugger,cursorPredictor 

initiateUserConfirmToProceed(forBreak=True)


# --- Decoder B ---
print("--- RUNNING DECODER TRIAL B --- DURATION: 3 MINUTES --- \n")

# Set Test game save Locations
decoderBSaveLocation = saveGameLocation + "_usingDecoderB.npz"
decoderBSaveLocationPkl = saveGameLocation + "_usingDecoderB.pkl"
decoderBMdlLocation = saveGameLocation + '_linearRigidBodyBModel.npz'

initiateUserConfirmToProceed()
gameEngine_decoderB = gameEngine
gameEngine_decoderB = configureForDecoder(gameEngine=gameEngine_decoderB,worldx=worldx,worldy=worldy,fps=fps,saveLocation=decoderBSaveLocation,
                saveLocationPkl=decoderBSaveLocationPkl,decoder= 'B', decoderLocation = decoderBMdlLocation)

player,targetBox,gameEngine_decoderB, clock, player_list,debugger,cursorPredictor =runSetup(gameEngine=gameEngine_decoderB)
runGame(gameEngine=gameEngine_decoderB, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list,clock = clock)

# check if data saved an inform user if game has not saved
verifySaved_decoderB = verifyGameSaveData(decoderBSaveLocation,decoderBSaveLocationPkl)
informUserGameSave(verifySaved=verifySaved_decoderB )

# Delete data after saved
del player,targetBox,gameEngine_decoderB, clock, player_list,debugger,cursorPredictor 

initiateUserConfirmToProceed(forBreak=True)

# --- Decoder C ---
print("--- RUNNING DECODER TRIAL C --- DURATION: 3 MINUTES --- \n")

# Set Test game save Locations
decoderCSaveLocation = saveGameLocation + "_usingDecoderC.npz"
decoderCSaveLocationPkl = saveGameLocation + "_usingDecoderC.pkl"
decoderCMdlLocation = saveGameLocation + '_linearRigidBodyCModel.npz'

initiateUserConfirmToProceed()
gameEngine_decoderC = gameEngine
gameEngine_decoderC = configureForDecoder(gameEngine=gameEngine_decoderC,worldx=worldx,worldy=worldy,fps=fps,saveLocation=decoderCSaveLocation,
                saveLocationPkl=decoderCSaveLocationPkl,decoder= 'C', decoderLocation = decoderCMdlLocation)

player,targetBox,gameEngine_decoderC, clock, player_list,debugger,cursorPredictor =runSetup(gameEngine=gameEngine_decoderC)
runGame(gameEngine=gameEngine_decoderC, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list,clock = clock)

# check if data saved an inform user if game has not saved
verifySaved_decoderC = verifyGameSaveData(decoderCSaveLocation,decoderCSaveLocationPkl)
informUserGameSave(verifySaved=verifySaved_decoderC )

# Delete data after saved
del player,targetBox,gameEngine_decoderC, clock, player_list,debugger,cursorPredictor 

initiateUserConfirmToProceed(forBreak=True)


# --- Decoder D ---
print("--- RUNNING DECODER TRIAL D --- DURATION: 3 MINUTES --- \n")

# Set Test game save Locations
decoderDSaveLocation = saveGameLocation + "_usingDecoderD.npz"
decoderDSaveLocationPkl = saveGameLocation + "_usingDecoderD.pkl"
decoderDMdlLocation = saveGameLocation + '_linearRigidBodyDModel.npz'

initiateUserConfirmToProceed()
gameEngine_decoderD = gameEngine
gameEngine_decoderD = configureForDecoder(gameEngine=gameEngine_decoderD,worldx=worldx,worldy=worldy,fps=fps,saveLocation=decoderDSaveLocation,
                saveLocationPkl=decoderDSaveLocationPkl,decoder= 'D', decoderLocation = decoderDMdlLocation)

player,targetBox,gameEngine_decoderD, clock, player_list,debugger,cursorPredictor =runSetup(gameEngine=gameEngine_decoderD)
runGame(gameEngine=gameEngine_decoderD, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list,clock = clock)

# check if data saved an inform user if game has not saved
verifySaved_decoderD = verifyGameSaveData(decoderDSaveLocation,decoderDSaveLocationPkl)
informUserGameSave(verifySaved=verifySaved_decoderD )

# Delete data after saved
del player,targetBox,gameEngine_decoderD, clock, player_list,debugger,cursorPredictor 

initiateUserConfirmToProceed(forBreak=True)

print("Experimental protocol has concluded")

"""
PHASE 3 END
"""



"""
    PHASE 4: Post data analysis 
"""
print("--- PHASE 4: Testing out decoders in the closed loop ---")