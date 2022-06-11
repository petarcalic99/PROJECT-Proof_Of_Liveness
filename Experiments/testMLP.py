import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



n_epochs = 6
batch_size_train = 64
batch_size_test = 1000
learning_rate = 1
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root='./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.RandomAffine(degrees = (-0,0),translate=(0.3,0.3)),  
                               torchvision.transforms.ToTensor(),
                               
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root='./data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.RandomAffine(degrees = (-0,0),translate=(0.3,0.3)),  
                               torchvision.transforms.ToTensor(),
                               
                             ])),
  batch_size=batch_size_test, shuffle=True)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32,10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)   # Skip for inference
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


model = Net()
#model.load_state_dict(torch.load('mnist_dnn.pt'))
optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)


print("the model")
print(model)



def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            


    
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))






test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()



"""
around 5 epochs and lr = 1
Training with no translation : fast learning and ok score of 94%
Training with no translation and test on 0.1 translation: 59%
Training with no translation and smaller learning rate of 0.01 on test with 0.2 translation :21%
Training with no translation and test on translation 0.2: 25%
Training on translation 0.1 and test on no translation: 92% 
Training on translation 0.2 and test on no translation: 80%
Training on translation 0.1 and test on 0.1: 88%   !!! feasable results if we ask the user to do another test
Training on translation 0.1 and test on 0.2:  54%
Training on tranlsation 0.2 and test on 0.2: 77%
Training on translation 0.2 and test on 0.1: 81%
Training on translation 0.3 and test on 0: 66%
Training on translation 0.3 and test on 0.1: 68%
Training on translation 0.3 and test on 0.1: 70%


"""



