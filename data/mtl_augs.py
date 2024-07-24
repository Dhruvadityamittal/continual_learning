from typing import Any
from data.data_transformation import DA_Permutation
import numpy as np

class PermuteAxis(object):
    def __init__(self, nPerm=4, minSegLength=10):
        self.nPerm = nPerm
        self.minSegLength = minSegLength

    # Input is (samples, 3)    
    def __call__(self, sample):
        sample = DA_Permutation(sample, self.nPerm, self.minSegLength)
        return sample

class FlipAxis(object):
    # Input is (samples, 3)
    def __call__(self, sample) -> Any:
        sample = np.flip(sample, 0)
        return sample