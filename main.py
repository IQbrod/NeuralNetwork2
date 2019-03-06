#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch as tor
import torchvision
import torchvision.transforms as transforms

from pydoc import locate
import matplotlib.pyplot as plt
import numpy as np
import time

import torch.nn as nn
import torch.optim as optim
import os
import sys

from modelstat import NNStat


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

#Definition des classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#Fonction d'affichage d'image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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

    # -- operation in F3
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
name = "v1"
Net = locate('models.'+name+'.Net')

net = Net()
stat = NNStat()
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
acc = stat.accuracy(net)

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
    acc = stat.accuracy(net)

print('Finished Training\n')

# --- Statistiques de l'apprentissage ---
print("Taux d'erreur global:", str(100-acc) +"%")
print("Nombre d'operations effectuées:",nbop * nb_epoch)
print("Temps d'execution (s):", round(timeTotal,2))
print("Operation / seconde:", (nbop*nb_epoch) / timeTotal)

# --- Save model
model_file = "models/"+name+".pt"
if not os.path.isfile(model_file):
    tor.save(net,model_file)
else:
    tor.save(net,"models/temp.pt")