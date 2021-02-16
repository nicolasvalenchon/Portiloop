# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:00:21 2021

@author: Nicolas
"""
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from time import time
from pathlib import Path
import os

t_start = time()
path_dataset = Path(__file__).absolute().parent.parent / 'dataset'
os.chdir(path_dataset)
signal_files = glob.glob("*_from_portiloop.txt")
fe = 250

signal_list = [scale(pd.read_csv(file, header=None)) for file in signal_files]

signal = np.hstack(signal_list)

np.savetxt("test_portiloop_data.txt", signal, fmt='%e')
print("tot_time = ", str(time() - t_start))