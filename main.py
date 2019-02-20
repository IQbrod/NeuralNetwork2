#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch as tor
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#Definition des tailles
train_size = 4
test_size = 4
nb_epoch = 2

#Creation du tensor
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

#Import du jeu de donnée (pour Training)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#Entraineur
trainLoader = tor.utils.data.DataLoader(trainset, batch_size= train_size, shuffle=True, num_workers=2)

#Import d'un jeu de test (sans Training)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#Testeur
testloader = tor.utils.data.DataLoader(testset, batch_size= test_size, shuffle=False, num_workers=2)

#Definition des classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#Fonction d'affichage d'image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
#dataiter = iter(trainLoader)
#images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(train_size)))

#Function to test accuracy on the entire dataset
def test_nn_accuracy(nt,dataset):
    correct = 0
    total = 0
    with tor.no_grad():
        for data in dataset:
            images, labels = data
            outputs = nt(images)
            _, predicted = tor.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('\033[1mAccuracy: %d %%\033[0m' % (100 * correct / total))
    return (100 * correct / total)

def op_counter(nt,data,display):
    inpt = data[0][0]
    par = list(nt.parameters())
    nb_conn_oneimg = 0
    nb_red_oneimg = 0

    nBatch = len(data)
    sBatch = train_size
    #sBatch = inpt.size()[0]

    # -- Operation in conv1 --
    xIn = inpt.size()[1]
    yIn = inpt.size()[2]
    zIn = inpt.size()[0]
    zOut = par[0].size()[0]
    #zIn = par[0].size()[1]
    xFil = par[0].size()[2]
    yFil = par[0].size()[3]

    xOut = xIn - (xFil-1) #Conv2d Padding
    yOut = yIn - (yFil-1) #Conv2d Padding

    nb_conn_oneimg += xOut * yOut * zOut * xFil * yFil * zIn

    # -- Operation in pool1
    pool_size = 2
    xOut = (xOut/pool_size)
    yOut = (yOut/pool_size)
    nb_red_oneimg += xOut * yOut * zOut

    # -- Operation in conv2
    xIn = xOut
    yIn = yOut
    zIn = zOut

    xFil = par[2].size()[2]
    xFil = par[2].size()[3]

    xOut = xIn - (xFil-1) #Conv2d Padding
    yOut = yIn - (yFil-1) #Conv2d Padding
    zOut = par[2].size()[0]

    nb_conn_oneimg += xOut * yOut * zOut * xFil * yFil * zIn

    # -- Operation in pool2
    xOut = (xOut/pool_size)
    yOut = (yOut/pool_size)

    nb_red_oneimg += xOut * yOut * zOut

    # -- Operation in F1
    sIn = xOut * yOut * zOut
    #sIn = par[4].size()[1]
    sOut = par[4].size()[0]

    nb_conn_oneimg += sIn * sOut

    # -- operation in F2
    sIn = sOut
    #sIn = par[6].size()[1]
    sOut = par[6].size()[0]
    
    nb_conn_oneimg += sIn * sOut

    # -- operation in F2
    sIn = sOut
    #sIn = par[8].size()[1]
    sOut = par[8].size()[0]
    
    nb_conn_oneimg += sIn * sOut
    
    if (display):
        nb_conn = int(nb_conn_oneimg * sBatch * nBatch)
        nb_red = int(nb_red_oneimg * sBatch * nBatch)
        total = 2*nb_conn + 3*nb_red
        print("Operations per epoch on the entire dataset")
        print("Addition:",nb_conn)
        print("Multiplication:", nb_conn)
        print("Maximum:", nb_red * 3)
        print("Total:", total)

    return total

#Creation du réseau
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
#Creation de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# 1. Couches et sous-couches
print(net)
# 2. Tailles des tenseurs Xn et du poids Wn/Bn
params = list(net.parameters())
i = 0
while i < len(params):
    print("W"+ str(i//2) +": "+ str(params[i].size()))
    print("B"+ str(i//2) +": "+ str(params[i+1].size()))
    i += 2
print()

# Nombre d'operations du réseau sur les images
nbop = op_counter(net,trainset,1)
print()

# 3. Affichage pour chaque époque
print("== Before Training ==")
acc = test_nn_accuracy(net,testloader)

# --- Training du réseau ---
timeTotal = 0
for epoch in range(nb_epoch):  # loop over the dataset multiple times
    print("== Epoch "+str(epoch+1)+" ==")
    running_loss = 0.0
    timeStart = time.time()
    for i, data in enumerate(trainLoader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        #running_loss += loss.item()
        #if i % 2000 == 1999:
        #    print('[%d, %5d] loss: %.3f' %
        #          (epoch + 1, i + 1, running_loss / 2000))
        #    running_loss = 0.0

    # 3. Accuracy per epoch
    timeTotal += time.time() - timeStart
    acc = test_nn_accuracy(net,testloader)

print('Finished Training')

# --- Test du réseau ---
# Récuperation du jeu de test
dataiter = iter(testloader)
images, labels = dataiter.next()

# Prediction des sorties
outputs = net(images)

# Recuperation de classe par l'energie
_, predicted = tor.max(outputs, 1)

# Affichage [Verité]: Prédiction
print()
for j in range(4):
    sym = "\033[32m✓\033[0m" if labels[j] == predicted[j] else "\033[31m✗\033[0m"
    print("["+classes[labels[j]]+"]: "+classes[predicted[j]]+ " "+sym)
print()

# --- Test par classe ---
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with tor.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = tor.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
print()

# --- Statistiques de l'apprentissage ---
print("Taux d'erreur global:", 100-acc)
print("Nombre d'operations effectuées:",nbop * nb_epoch)
print("Temps d'execution (s):", timeTotal)
print("Operation / seconde:", (nbop*nb_epoch) / timeTotal)