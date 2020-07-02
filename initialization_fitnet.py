import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import sys

################################# network ######################################

class FitNet_1(nn.Module):
    def __init__(self):
        super(FitNet_1, self).__init__()
        # Input: 32x32x3
        self.conv1 = nn.Conv2d( 3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        # Input: 16x16x16
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Input: 8x8x32
        self.conv7 = nn.Conv2d(32, 48, 3, padding=1)
        self.conv8 = nn.Conv2d(48, 48, 3, padding=1)
        self.conv9 = nn.Conv2d(48, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(8, 8)
        # Input: 1x1x64
        self.fc1 = nn.Linear(64, 10)

        # save the trainable layers in a list for initializing
        self.hidden = [self.conv1, self.conv2, self.conv3, \
                       self.conv4, self.conv5, self.conv6, \
                       self.conv7, self.conv8, self.conv9, \
                       self.fc1]


    def forward(self, x):
        # forward pass through the network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool1(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool2(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool3(x)

        x = x.view(-1, 64)

        x = self.fc1(x)
        x = F.softmax(x, dim=0)

        return x


    def initialize_uniform(self, u=(-0.01,0.01), c=1.0):
        for layer in self.hidden:
            nn.init.uniform_(layer.weight, a=u[0], b=u[1])
            nn.init.constant_(layer.bias, c)


    def initialize_normal(self, mean=0.0, std=0.01, c=1.0):
        for layer in self.hidden:
            nn.init.normal_(layer.weight, mean=mean, std=std)
            nn.init.constant_(layer.bias, c)


    def initialize_xavier_u(self):
        for layer in self.hidden:
            nn.init.xavier_uniform_(layer.weight, 
                           gain=nn.init.calculate_gain('relu', param=None))
            nn.init.constant_(layer.bias, 0.0)

    def initialize_xavier_n(self):
        for layer in self.hidden:
            nn.init.xavier_normal_(layer.weight, 
                           gain=nn.init.calculate_gain('relu', param=None))
            nn.init.constant_(layer.bias, 0.0)


    def initialize_kaiming_u(self):
        for layer in self.hidden:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.0)


    def initialize_kaiming_n(self):
        for layer in self.hidden:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.0)



    def print_wb(self, layer_list=None):
        '''
        Prints the parameter values of the specified layers (subset of[1,...,L]).
        '''
        torch.set_printoptions(precision=16)
        if layer_list is None:
            layer_list  = list(range(1,len(self.hidden)))
        else:
            layer_list = list(layer_list)
            max_val    = max(layer_list)
            min_val    = min(layer_list)
            assert max_val <= (len(self.hidden)-1)
            assert min_val >= 1

        for idx in layer_list:
            print('weight of layer '+repr(idx)+' = '+repr(self.hidden[idx-1].weight))
            print('bias   of layer '+repr(idx)+' = '+repr(self.hidden[idx-1].bias))
            print('\n')


    def save_state_dict(self, PATH):
        '''
        Saves the current model state.
        A common PyTorch convention is to save models using either a .pt or 
        .pth file extension.
        Input:
            -PATH: absolute path as string
        '''
        torch.save(self.state_dict(), PATH)


    def save_model(self, PATH):
        '''
        Saves the current model.
        A common PyTorch convention is to save models using either a .pt or 
        .pth file extension.
        Input:
            -PATH: absolute path as string
        '''
        torch.save(self, PATH)


    @classmethod
    def load_state_dict_(cls, PATH):
        '''
        Loads a state dict of a nn-model of cls. The last _ is needed there, 
        because otherwise we would override the pytorch load_state_dict method.
        Input:
            -PATH: absolute path as string
        '''

        nn_model = cls()
        nn_model.load_state_dict(torch.load(PATH))

        return nn_model


    @classmethod
    def load_model(cls, PATH):
        '''
        Loads a nn-model of cls.
        Input:
            -PATH: absolute path as string
        '''
        nn_model = torch.load(PATH)

        return nn_model


#################################### data ######################################

# load the data and transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


################################## training ####################################

# get the network
fitnet1 = FitNet_1()
# initialize the model
#fitnet1.initialize_xavier_u()

#fitnet1.initialize_xavier_n()
#
#fitnet1.initialize_kaiming_u()
#
fitnet1.initialize_kaiming_n()

#fitnet1.save_state_dict('./fitnet1.pt')
#fitnet1.save_model('./fitnet1_model.ptr')

# define the lossfunction
loss_func = torch.nn.CrossEntropyLoss()
# get the optimizer
optimizer = optim.SGD(fitnet1.parameters(), lr=0.001, momentum=0.9)

# train the network
k = 0
for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        nn_outputs = fitnet1(inputs)
        loss = loss_func(nn_outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 12000 == 11999:    # print every 1000 mini-batches
            k += 1
            print('[%d, %5d] loss: %.4f iteration %5d' %
                  (epoch + 1, i + 1, running_loss / 12000, k))
            running_loss = 0.0

print('Finished Training')
# save the model
fitnet1.save_model('./fitnet1_trained_kaiming_normal.ptr')


############################### post processing ################################
# check the accuracy of the network
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = fitnet1(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
      100 * correct / total))


# check the accuracy per class of the network
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = fitnet1(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))



