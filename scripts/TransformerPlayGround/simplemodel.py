import torch
import torch.nn as nn

class simpleModel(nn.Module): # inheriting from nn.module
    def __init__(self):
        super(simpleModel,self).__init__()
        self.fc1=nn.Linear(10,5)
        self.fc2= nn.linear(5,2)
        
    def forward(self,x):
        x= self.fc1(x) # pass the input x through the linear layer
        x= torch.relu(x) # apply a non-linear activation function (Relu)
        x=self.fc2(x)
        return x
model= simpleModel()
input_data  =  torch.randn(5,10)
output = model(input_data)
print(output)