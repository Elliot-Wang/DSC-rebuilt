
import scipy.io as sio
from os.path import dirname, join as pjoin
import os

mat_file = pjoin(dirname(sio.__file__), 'Data', 'YaleBCrop025.mat')
data = sio.loadmat('DSC/Data/YaleBCrop025.mat')
with open('DSC/README.md') as f:
    pass

