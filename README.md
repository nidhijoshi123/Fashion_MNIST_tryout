# Fashion_MNIST_tryout

Playing around with Fashion MNIST dataset

CLASSIFICATION PROBLEM

1. The FashionMNIST dataset consisting of 60000 training images and 10000 test images has been used in this project. Similar to the MNIST digit dataset, the Fashion MNIST dataset includes:

60,000 training examples,
10,000 testing examples,
10 classes,
28×28 grayscale/single channel images.

2. The 10 underlying class labels (or categories) are:


0:T-shirt/top
1:Trouser/pants
2:Pullover shirt
3:Dress
4:Coat
5:Sandal
6:Shirt
7:Sneaker
8:Bag
9:Ankle boot

3. We train our CNN model on the Fashion MNIST dataset, evaluate it, and review the results. Pytorch deep learning framework has been used. Torchvision is used to handle the datasets.

4. Data_preparation.py gets us started with loading the training and test dataset. We perform the ETL (Extract,Transform and Loading) operations on our dataset here.

5. Main_model.py has the network definition. Forward pass, back prop, training the neural network are done here. Additionally, classification metrics (accuracy,loss) along with the real time tensorboard visualizations are defined here. Finally, the model is saved (model/model.pt).

6. Tensorboard run logs are saved in the 'runs' folder.

7. Confusion_matrix.py produces confusion matrix of the training dataset.

8. Testing_testSET produces confusion matrix of test dataset.

9. Supporting_functions has some functions for quick access.

10. Use fmnist_environment.yaml to set up the same conda environment (with all dependencies) as mine!
