# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch  
import Data_preparation
import Supporting_functions
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import torchvision

torch.set_printoptions(linewidth=120) #Display option for output
torch.set_grad_enabled(True) 

#Extend the super class nn.Module

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        
        #Add convolutional layers
        self.conv1= nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2= nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)
        
        #Add linear/fully connected layers
        self.fc1=nn.Linear(in_features=12*4*4,out_features=120)
        self.fc2=nn.Linear(in_features=120,out_features=60)
        self.out=nn.Linear(in_features=60,out_features=10)
        
        
    def forward(self,t):
            #forward pass
            #input layer
            
            t=t
            
            #first conv layer
            
            t=self.conv1(t)
            t=F.relu(t)
            t=F.max_pool2d(t,kernel_size=2,stride=2)
            
            #second conv layer
            
            t=self.conv2(t)
            t=F.relu(t)
            t=F.max_pool2d(t,kernel_size=2,stride=2)
            
            #first linear layer
            
            t=t.reshape(-1,12*4*4)
            t=self.fc1(t)
            t=F.relu(t)
            
            #second linear layer
            
            t=self.fc2(t)
            t=F.relu(t)
            
            #output layer
            
            t=self.out(t)
            #t=F.softmax(t,dim=1)
            
            
            return t


#Calculates total number of correct predictions 
#def get_num_correct(preds,labels):
 #    return preds.argmax(dim=1).eq(labels).sum().item()
       
#torch.set_grad_enabled(False) 

#ACCESSING A SINGLE IMAGE FROM TRAINING SET
#SAME CAN BE DONE FOR A BATCH OF IMAGES FROM DATA LOADER

#sample=next(iter(Data_preparation.train_set))
#image,label=sample
#print(image.shape)

#Create an instance of the Network class   
    
network=Network()

#Optimize the weights 

optimizer=optim.Adam(network.parameters(),lr=0.01)

#MAKING PREDICTION ON SINGLE IMAGE USING AN UN-TRAINED NETWORK
#NETWORK GUESSES RANDOMLY HERE BASED ON RANDOM WEIGHTS

#image.unsqueeze(0).shape                  #gives a batch with size 1
#pred=network(image.unsqueeze(0))
#print(pred)
#print(pred.argmax(dim=1))
#print(F.softmax(pred,dim=1))

images,labels=next(iter(Data_preparation.train_loader))
grid=torchvision.utils.make_grid(images)

tb=SummaryWriter()
tb.add_image('images',grid)
tb.add_graph(network,images)


for epoch in range(10):
    total_loss=0
    total_correct=0
    
    #Load images in batches of 100
    
    #batch=next(iter(Data_preparation.train_loader))
    for batch in Data_preparation.train_loader:
        images,labels=batch
        
        
        #Training and loss calculation
        
        preds=network(images)
        loss=F.cross_entropy(preds,labels)
        
        #Calculating Gradients
        optimizer.zero_grad()  #because we don't want to accumalate weights
        loss.backward()
        #print(network.conv1.weight.grad.shape)
        
        
        
        #Update the weights
        
        optimizer.step()
        
        total_loss+=loss.item()
        total_correct+=Supporting_functions.get_num_correct(preds,labels)
        
    #Tensorboard logs
    tb.add_scalar("Loss: ", total_loss,epoch)
    tb.add_scalar("Number of correct predictions: : ", total_correct,epoch)
    tb.add_scalar("Accuracy: ", total_correct/len(Data_preparation.train_set),epoch)
    
    tb.add_histogram("conv1 bias",network.conv1.bias,epoch)
    tb.add_histogram("conv1 weight",network.conv1.weight,epoch)
    tb.add_histogram("conv1 weight gradient",network.conv1.weight.grad,epoch)
    
    
    print("epoch: ",epoch,"total correct: ",total_correct,"total loss: ",total_loss)
    
accuracy=total_correct/len(Data_preparation.train_set)
print("Network Accuracy: ",accuracy*100,"%")

torch.save(network, 'C:/Users/nidhi/FashionMNIST/model/model.pt')