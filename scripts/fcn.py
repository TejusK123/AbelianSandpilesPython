import torch 
import torch.nn as nn
import numpy as np




test = np.load('recurrenceclassification.npz', allow_pickle= True)

test = test['arr_0']

targets = [item[1] for item in test]




print(len(targets))







