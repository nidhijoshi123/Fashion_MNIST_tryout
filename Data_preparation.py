# -*- coding: utf-8 -*-

import torch                                    #top level pytorch package and tensor library
import torchvision                              #for pytorch's computer-vision applications
import torchvision.transforms as transforms     #transformations for image processing

#Data Preparation (ETL)

#Extract(E) and Transform(T) image data into tensor data(Train dataset)

train_set=torchvision.datasets.FashionMNIST(
        root= './data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
        )

#Data loader(L) gives querying capabilities during NN training
#Querying capabilities: data shuffling, modifying the batch size

train_loader=torch.utils.data.DataLoader(train_set,batch_size=100, shuffle=True)

#Load test dataset
test_set=torchvision.datasets.FashionMNIST(
        root= './data/FashionMNIST',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
        )

test_loader=torch.utils.data.DataLoader(test_set)
