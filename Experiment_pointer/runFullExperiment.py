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

"""
END OF PHASE 2

"""

"""
PHASE 3: Testing decoders in the closed loop
# """
print("--- PHASE 3: Testing out decoders in the closed loop ---")

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