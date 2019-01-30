#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch as tor
import torchvision
import torchvision.transforms as transforms

#Creation du tensor
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

#Import du jeu de donnée (pour Training)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#Entraineur: taille du jeu 5 par 5
trainLoader = tor.utils.data.DataLoader(trainset, batch_size=5, shuffle=True, num_workers=2)

#Import d'un jeu de test (sans Training)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#Testeur: taille du jeu 3 par 3
testloader = tor.utils.data.DataLoader(testset, batch_size=3, shuffle=False, num_workers=2)

#Definition des classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')