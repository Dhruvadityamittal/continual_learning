from .cub import CUBirds
from .mit import MITs
from .dog import Dogs
from .air import Airs
from .ADL import  ADL_10s
from .Wisdm import  Wisdm
from .Wisdm import  Wisdm
from .import utils
from .realworld import realworld
from .oppo import oppo
from .pamap import pamap
from .mhealth import mhealth


_type = {
    'cub': CUBirds,
    'mit': MITs,
    'dog': Dogs,
    'air': Airs,
    'adl' :ADL_10s,
    'wisdm': Wisdm,
    'realworld' : realworld,
    'oppo' : oppo, 
    'pamap': pamap,
    'mhealth': mhealth
}

def load(name, root, mode, windowlen, transform=None, autoencoderType = None, standardize = None, fold = None):
    return _type[name](root=root, mode=mode, windowlen=windowlen, transform=transform, autoencoderType = autoencoderType, standardize = standardize, fold = fold)


            