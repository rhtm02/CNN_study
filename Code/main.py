# this file is main file
from CNN import *
import CNN
import DataLoader
import torch

# set hyperparameters
EPOCH = 10
LEARNING_RATE = 1e-3
KERNALSIZE = (5,5)

if (__name__ == "__main__"):
    #model definition
    model = Network()
    #dataloader definotion
    trainloader, testlodaer = DataLoader.LoadData()
    #configure loss function
    criterion = torch.nn.CrossEntropyLoss().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #training
    for epoch in range(EPOCH):
        avg_cost = 0

        for i, data in enumerate(trainloader):
            # image is already size of (28x28), no reshape
            # label is not one-hot encoded
            x, y = data
            if (CNN.use_cuda):
                x = x.to("cuda")
                y = y.to("cuda")


            optimizer.zero_grad()
            hypothesis = model(x)
            cost = criterion(hypothesis, y)
            cost.backward()
            optimizer.step()
            avg_cost += cost / len(trainloader)

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


else:
    pass