# this file is main file

#from CNN_2layer import Network, use_cuda
from CNN import Network, use_cuda
#from CNN_maxpooling import Network, use_cuda
import DataLoader
import torch

torch.cuda.empty_cache()
# set hyperparameters
EPOCH = 10
LEARNING_RATE = 1e-3
KERNALSIZE = (5,5)
BATCHSIZE = 1

if (__name__ == "__main__"):
    #model definition
    model = Network()
    print(model)
    #dataloader definotion
    trainloader, testlodaer = DataLoader.LoadData(BatchSize = BATCHSIZE)
    #configure loss function
    criterion = torch.nn.CrossEntropyLoss().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #training
    for epoch in range(EPOCH):


        for i, data in enumerate(trainloader):
            # image is already size of (28x28), no reshape
            # label is not one-hot encoded
            x, y = data
            if (use_cuda):
                x = x.to("cuda")
                y = y.to("cuda")


            optimizer.zero_grad()
            hypothesis = model(x)
            cost = criterion(hypothesis, y)
            cost.backward()
            optimizer.step()


        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, cost))

    # save trained model

    #torch.save(model.state_dict())


    # test session
    with torch.no_grad():
        for i, data in enumerate(trainloader):
            X_test, Y_test = data
            if(use_cuda):
                X_test = X_test.to("cuda")
                Y_test = Y_test.to("cuda")
            prediction = model(X_test)
            correct_prediction = torch.argmax(prediction, 1) == Y_test
            accuracy = correct_prediction.float().mean()
            print('Accuracy:', accuracy.item())
else:
    pass