import numpy as np 
from ASBclassdef import Sandpile
from topple import stabilize, create_visualization
import matplotlib.pyplot as plt
with open(r'testimages\tinyshakespeare.txt', 'r') as file:
    data = file.read().replace('\n', ' ')



del file


asciidata = [bin(ord(i))[2:] for i in data]




testpile = stabilize(np.full((100,100), 6), data = False)[0]

