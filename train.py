# the training loop and connecting the data pipeline to model


import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


def training_loop(model:nn.Module,epochs:int,loss_fn,train_loader, test_loader, device ):
    training_loss, validation_loss = [], []
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps'
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',0.1,3)
    loss_fn = torch.nn.CrossEntropyLoss()
    print(f'device using {device}')
    train_acc,val_accuracy = 0
    for i in tqdm(range(epochs)):
        model.train()
        print('training...')
        for batch_num,(x,y) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred.argmax(dim=1),y)
            loss.backwards()
            optimizer.step()
            optimizer.zero_grad()
            training_loss.append(loss.item())

        print(f'Epoch :{epochs} | training loss:{loss.item()}')

        # validation
        model.eval()
        print('validating...')
        for batch_num, (x,y) in enumerate(test_loader):
            x,y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred.argmax(dim=1),y)
            lr_scheduler.step(loss)
            validation_loss.append(loss.item())


        print(f'Epoch :{epochs} | training loss:{loss.item()}')
        # if epochs % 10 ==0:
            # plt.plot(training_loss,epochs)
            # plt.plot(validation_loss,epochs)


