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
train_size = 16
test_size = 4
nb_epoch = 12

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

#Creation du réseau
name = "v7"
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
nbop = net.op_counter(trainset,1)
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