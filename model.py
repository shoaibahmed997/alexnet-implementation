import torch
import torch.nn as nn


class Alexnet(nn.Module):
    """ the net contains eight layers with weights;
    the first five are convolutional and the remaining three are fullyconnected. 
    The output of the last fully-connected layer is fed to a 87-way softmax which produces
    a distribution over the 87 class labels.
    """

    def __init__(self, channels:int, classes:int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.lrn = nn.LocalResponseNorm(5,k=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(in_features= 256*6*6, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.output = nn.Linear(in_features=4096, out_features=classes)
        self.dropout = nn.Dropout(0.5)
        self.init_weight()

    def forward(self,x):
        x = self.max_pool(self.lrn(self.relu(self.conv1(x))))
        x = self.max_pool(self.lrn(self.relu(self.conv2(x))))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.max_pool(self.lrn(self.conv5(x)))
        x = torch.flatten(x,1)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.output(x)
        return x
    
    def init_weight(self):
        bias =[2,4,5,9,10,11]
        for i,layer in enumerate(self.modules()):
            if  isinstance(layer,nn.Conv2d) or  isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight,mean=0,std=0.01)
                if i in bias:
                    nn.init.constant_(layer.bias,1)
                else:
                    nn.init.constant_(layer.bias,0)