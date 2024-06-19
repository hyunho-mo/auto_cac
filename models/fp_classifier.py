'''
Implementation of CNN classifying false positive patchs
'''

# PyTorch libraries 
import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import random
import numpy as np

r_state = 0
random.seed(r_state)
np.random.seed(r_state)
torch.manual_seed(r_state)

# # Define the model
# class VGG_Net(Module):   
#     def __init__(self):
#         super(VGG_Net, self).__init__()

#         self.cnn_layers = Sequential(
#             Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             ReLU(inplace=True),
#             BatchNorm2d(32),
#             MaxPool2d(kernel_size=2, stride=2),
#             Dropout(p=0.25),
#             Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             ReLU(inplace=True),
#             BatchNorm2d(64),
#             MaxPool2d(kernel_size=2, stride=2),
#             Dropout(p=0.25),
#             Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             ReLU(inplace=True),
#             BatchNorm2d(128),
#             MaxPool2d(kernel_size=2, stride=2),
#             Dropout(p=0.25),
#             Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             ReLU(inplace=True),
#             BatchNorm2d(128),
#             MaxPool2d(kernel_size=2, stride=2),
#             Dropout(p=0.25),
#         )

#         self.linear_layers = Sequential(
#             Linear(128 * 2 * 2, 512),
#             ReLU(inplace=True),
#             Dropout(),
#             Linear(512, 256),
#             ReLU(inplace=True),
#             Dropout(),
#             Linear(256,10),
#             ReLU(inplace=True),
#             Dropout(),
#             Linear(10,2)
#         )

#     # Defining the forward pass    
#     def forward(self, x):
#         x = self.cnn_layers(x)
#         print ("x.shape", x.shape)
#         x = x.view(x.size(0), -1)
#         print ("x.shape", x.shape)
#         x = self.linear_layers(x)
#         return x


# Define the model
class VGG_Net(Module):   
    def __init__(self, in_channel = 3):
        super(VGG_Net, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(32),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=0.25),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(64),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=0.25),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(128),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=0.25),
        )

        self.linear_layers = Sequential(
            Linear(128 * 5 * 5, 512),
            ReLU(inplace=True),
            Dropout(),
            Linear(512, 256),
            ReLU(inplace=True),
            Dropout(),
            Linear(256,10),
            ReLU(inplace=True),
            Dropout(),
            Linear(10,2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        # print ("x.shape", x.shape)
        x = x.view(x.size(0), -1)
        # print ("x.shape", x.shape)
        x = self.linear_layers(x)
        return x


# CNN based on U-net encoder architecture
    