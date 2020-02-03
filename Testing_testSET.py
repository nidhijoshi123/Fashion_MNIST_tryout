# -*- coding: utf-8 -*-

import Data_preparation
#import Main_model
import Supporting_functions
import torch
from sklearn.metrics import confusion_matrix


model = torch.load('C:/Users/nidhi/FashionMNIST/model/model.pt')

prediction_loader=torch.utils.data.DataLoader(Data_preparation.test_set,batch_size=10000)
test_preds=Supporting_functions.get_all_preds(model,prediction_loader)

with torch.no_grad():
    prediction_loader= torch.utils.data.DataLoader(Data_preparation.test_set,batch_size=10000)
    test_preds=Supporting_functions.get_all_preds(model,prediction_loader)
    
preds_correct= Supporting_functions.get_num_correct(test_preds,Data_preparation.test_set.targets)

print("total correct predictions (test set): ",preds_correct)
print("Accuracy (test set): ", (preds_correct/len(Data_preparation.test_set))*100,"%")

#Confusion matrix sklearn approach

cm=confusion_matrix(Data_preparation.test_set.targets,test_preds.argmax(dim=1))
print("Confusion matrix sklearn approach (test set): \n", cm)