# this file is CNN network configuration
import torch
import torch.nn as nn
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
#batch norm
class Network(nn.Module):

    def __init__(self, kernalsize = (4,4)):
        super(Network, self).__init__()
        #convelution layer
        self.conv_module = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, stride=1, kernel_size=kernalsize),
            #print("C1"),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 2),
            nn.Conv2d(in_channels=2, out_channels=4, stride=1, kernel_size=kernalsize, padding=(3, 3)),
            #print("C2"),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 4),
            nn.Conv2d(in_channels=4, out_channels=8, stride=1, kernel_size=kernalsize, padding=(3, 3)),
            #print("C3"),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 8),
            nn.Conv2d(in_channels=8, out_channels=16, stride=1, kernel_size=kernalsize),
            #print("C4"),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 16)
        )
        # Affine layer
        self.affine_module = nn.Sequential(
            nn.Linear(in_features = 16*28*28, out_features = 1200),
            nn.ReLU(),
            nn.Linear(in_features=1200, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10)
        )


        if (use_cuda):
            self.conv_module = self.conv_module.cuda()
            self.affine_module  =self.affine_module.cuda()
            print("USE CUDA")

    def forward(self, x):

        out = self.conv_module(x)
        #print("Conv output : " + str(out.shape))
        #flatten
        dim = 1
        for d in out.size()[1:]:
            dim = dim * d
        out = out.view(-1, dim)

        out = self.affine_module(out)

        return F.softmax(out, dim = 1)



if ( __name__ == "__main__"):
    print("CNN module")

else:
    pass
    #print("import CNN module")
