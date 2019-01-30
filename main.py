#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch as tor
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


#Definition des tailles
train_size = 30
test_size = 5

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
dataiter = iter(trainLoader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(train_size)))