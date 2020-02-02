# -*- coding: utf-8 -*-

import Main_model
import Data_preparation
import torch 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix 

#Makes predictions on batch of images and concatenates

def get_all_preds(model,loader):
    all_preds=torch.tensor([])
    for batch in loader:
        images,labels=batch
        
        preds=model(images)
        all_preds=torch.cat((all_preds,preds),dim=0)
        return all_preds
    
    
prediction_loader=torch.utils.data.DataLoader(Data_preparation.train_set,batch_size=60000)

#Network predictions

train_preds=get_all_preds(Main_model.network,prediction_loader)

#Locally turn off gradient tracking for memory management
#Don't create gradient graphs for making predictions

with torch.no_grad():
    prediction_loader= torch.utils.data.DataLoader(Data_preparation.train_set,batch_size=60000)
    train_preds=get_all_preds(Main_model.network,prediction_loader)
    
preds_correct= Main_model.get_num_correct(train_preds,Data_preparation.train_set.targets)

print("total correct: ",preds_correct)
print("Accuracy: ", preds_correct/len(Data_preparation.train_set),"%")

#Confusion matrix traditional approach

stacked= torch.stack(
        (Data_preparation.train_set.targets,train_preds.argmax(dim=1)),
        dim=1)

cmt=torch.zeros(10,10,dtype=torch.int32)

for p in stacked:
    true_label,pred_label=p.tolist()
    cmt[true_label,pred_label]=cmt[true_label,pred_label]+1
    
print("Confusion matrix traditional approach: \n", cmt)

#Confusion matrix sklearn approach

cm=confusion_matrix(Data_preparation.train_set.targets,train_preds.argmax(dim=1))
print("Confusion matrix sklearn approach: \n", cm)

    


         