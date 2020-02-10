# this file is CNN network configuration
import torch.nn as nn

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        #inirialize convelution layer
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()
        self.conv3 = nn.Conv2d()
        self.conv4 = nn.Conv2d()

        #initialize pooling layer
        self.pooling = nn.MaxPool2d()

    def forward(self, x):



        return x















if ( __name__ == "__main__"):
    print("CNN module")
else:
    print("import CNN module")
