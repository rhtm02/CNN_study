# this file is DataLoader configuration script

import torch
import torchvision
import torchvision.transforms as transforms


def LoadData(BatchSize = 100):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(([0.5]), ([0.5]))])

    trainset = torchvision.datasets.FashionMNIST(root = '../Data/fashion_MNIST/', train = True, download = True, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BatchSize, shuffle = True, num_workers = 2)

    testset = torchvision.datasets.FashionMNIST(root = '../Data/fashion_MNIST/', train = False, download = True, transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size = BatchSize, shuffle = False, num_workers = 2)

    return trainloader, testloader


if ( __name__ == "__main__"):
    print("LoadData Module")

else:
    pass
    #print("LOADING Fashion-MNIST DATA")

