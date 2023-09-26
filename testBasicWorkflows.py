"""
test basic workflows
"""

# gather dependencies
import numpy as np
import sys

def testNumpyImported():
    assert "numpy" in sys.modules

testNumpyImported()