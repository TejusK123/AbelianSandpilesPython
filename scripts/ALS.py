import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from convlstm import *
from topple import stabilize, create_visualization

data = np.load('50x50topdata.npz')
testtarget = torch.tensor(data['arr_0'][:,:,1728:]).reshape(126, 50, 50).float()
testdata = data['arr_0'][:,:,:1728]
print(testtarget.shape)
print(testdata.shape)
testdata = torch.tensor(testdata.reshape(1, 1728,1, 50, 50))



class conv(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = ConvLSTM(1, 1, (3,3), 1, True)
        self.relu1 = nn.ReLU()
        

    def forward(self, x):
        
        _, x = self.conv(x)
        x = self.relu1(x[0][0])
        
        

        
        x = x.view(1, 50, 50)
        return(x)
    


model = conv()
lossf_n = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr = 0.01)
data = np.array([])
for i in range(50):

    optim.zero_grad()
    output = model(testdata)
    loss = lossf_n(output, testtarget)
    loss.backward()
    optim.step()
    if i % 10 == 0:
        print(f"{i}: {loss}")
        accuracy = torch.sum((torch.round(output) == testtarget)) / 2500
        data = np.append(data, accuracy.item())





import matplotlib.pyplot as plt
plt.plot(data)
plt.show()











