"""
test file to verify that offline game works 
"""

# import common libraries
import sys
import subprocess
import pytest

# add Root Directory to system path to import created packages
sys.path.insert(0,'/Users/rishitabanerjee/Desktop/BrainMachineInterfaces/')
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')

# import variables file where all game data is defined

# change some vars from the game engine




def testGameRunsErrorFree():
    from Experiment_pointer.variables import gameEngine
    from Experiment_pointer.setup import runSetup, endProgram
    import Experiment_pointer.runner as runner
    gameEngine.FETCHDATAFROMREALTIME = False
    gameEngine.recordData = False
    gameEngine.readData = False
    gameEngine.timeProgram = 30 # in seconds
    gameEngine.testMode = False
    # now run setup for the game
    player,targetBox,gameEngine, clock, player_list,debugger,player_list = runSetup(gameEngine=gameEngine)
    # don't set it up in test mode to verify it works
    
    with pytest.raises(SystemExit) as sample:
        runner.runGame(gameEngine=gameEngine, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list)

def testCursorMovesWhenKeypadUsed():
    pass

def testGameCanReadSimulatedData():
    from Experiment_pointer.variables import gameEngine
    from Experiment_pointer.setup import runSetup, endProgram
    import Experiment_pointer.runner as runner
    gameEngine.FETCHDATAFROMREALTIME = False
    gameEngine.recordData = False
    gameEngine.readData = True
    gameEngine.readLocation = 'PointerExperimentData/22_11_ashTrial1.npz'
    gameEngine.timeProgram = 30 # in seconds
    gameEngine.testMode = True
    # now run setup for the game
    player,targetBox,gameEngine, clock, player_list,debugger,player_list = runSetup(gameEngine=gameEngine)
    # set it up in test mode
    runner.runGame(gameEngine=gameEngine, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list)

def testDebugModeWorksOffline():
    from Experiment_pointer.variables import gameEngine
    from Experiment_pointer.setup import runSetup, endProgram
    import Experiment_pointer.runner as runner
    gameEngine.FETCHDATAFROMREALTIME = False
    gameEngine.recordData = False
    gameEngine.readData = False
    gameEngine.timeProgram = 10 # in seconds
    # now run setup for the game
    
    player,targetBox,gameEngine, clock, player_list,debugger,player_list = runSetup(gameEngine=gameEngine)
    # set it up in test mode
    outputVars = runner.runGame(gameEngine=gameEngine, player = player,debugger = debugger, targetBox = targetBox,player_list=player_list)
    print(outputVars)

def testRedBoxChangesGreenWhenHit():
    pass

def testOfflineDataCausesCorrectPointerMotion():
    pass


# add these to online
#testGameCanReadSimulatedData()
