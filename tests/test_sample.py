import pandas as pd
import numpy as np

def func(x):
    return x + 1

def test_answer():
    assert func(4) == 5
    
def test_np():
    print(np.array([2,3,4]))
    assert True==True