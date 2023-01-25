import torch
from torch import nn

class LocalTraining():

    """
    Base class for Local Training
    """

    def __init__(self, 
                 num_updates_in_epoch=None,
                 num_local_epochs=1):
       
        self.name = "local-training"
        self.num_updates = num_updates_in_epoch
        self.num_local_epochs = num_local_epochs
        

    def train(self, model, trainloader, criterion=None, opt=None, lr = 1e-2):
        """
        Method for local training
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if opt is None:
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        if self.num_updates is not None:
            self.num_local_epochs = 1

        model.train()
        running_loss = 0.0
        for epoch in range(self.num_local_epochs):
            for batch_id, (data, target) in enumerate(trainloader):
                x_batch, y_batch = data, target

                opt.zero_grad()

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                loss.backward()
                opt.step()
                
                running_loss += loss.item()

                if self.num_updates is not None and batch_id >= self.num_updates:
                    break

        return model, running_loss/(batch_id+1)
