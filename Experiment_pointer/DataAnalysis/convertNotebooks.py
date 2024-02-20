import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import NotebookExporter

import sys
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')
from Experiment_pointer.runFullExperiment import saveDirectory,saveGameLocation

# # Path to the notebook file
# input_notebook_path = 'Experiment_pointer/DataAnalysis/OPENLOOPpostExperimentDataProcessing.ipynb'
# output_notebook_path = saveGameLocation + 'Analysis-openLoopPostExperimentDataProcessing.ipynb'

# # Load the notebook
# with open(input_notebook_path) as f:
#     nb = nbformat.read(f, as_version=4)

# # Set up the ExecutePreprocessor
# ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

# # Execute the notebook
# ep.preprocess(nb, {'metadata': {'path': './'}})  # Specify the path for any relative paths in the notebook

# # Save the executed notebook
# exporter = NotebookExporter()
# body, _ = exporter.from_notebook_node(nb)

# with open(output_notebook_path, 'w') as f:
#     f.write(body)

# Path to the notebook file
notebookInputPath = 'Experiment_pointer/DataAnalysis/OPENLOOPpostExperimentDataProcessing.ipynb'
notebookOutputPath = saveGameLocation + 'Analysis-openLoopPostExperimentDataProcessing.ipynb'
def analyseNotebook(notebookInputPath, notebookOutputPath):

    

    # Load the notebook
    print("Opening Notebook ...")
    with open(notebookInputPath) as f:
        nb = nbformat.read(f, as_version=4)

    # Set up the ExecutePreprocessor
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    # Execute the notebook
    print("Executing Notebook ...")
    ep.preprocess(nb, {'metadata': {'path': './'}})  # Specify the path for any relative paths in the notebook

    # Save the executed notebook
    exporter = NotebookExporter()
    body, _ = exporter.from_notebook_node(nb)

    print("Exporting Notebook ...")
    with open(notebookOutputPath, 'w') as f:
        f.write(body)
    
    print("Done.")




analyseNotebook(notebookInputPath,notebookOutputPath)