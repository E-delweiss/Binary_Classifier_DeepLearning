#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 08:22:13 2021

@author: thierryk
"""

import numpy as np
import datetime
from PIL import Image

from import_data import load_data
import functions as fc
import models

data_train_orig, label_train, data_val_orig, label_val, classes = load_data()

m_train = data_train_orig.shape[0]
m_test = data_val_orig.shape[0]


# Reshape the training and test examples 
data_train_flatten = data_train_orig.reshape(m_train, -1).T
data_val_flatten = data_val_orig.reshape(m_test, -1).T

# Standardize data to have feature values between 0 and 1.
data_train = data_train_flatten/255.
data_val = data_val_flatten/255.

label_train = label_train.T
label_val = label_val.T

print ("data_train's shape: " + str(data_train.shape))
print ("data_val's shape: " + str(data_val.shape))
print ("\n----------------------\n")

n_x = data_train_orig.shape[1]*data_train_orig.shape[2]*data_train_orig.shape[3]
n_y = 1
layers_dims = [n_x, 20, 7, 5, n_y] #  4-layers model
learning_rate = 0.0075


def NN_model(X, Y, layers_dims, learning_rate = 0.0075, epochs = 1000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Parameters
    ----------
    X : np.array of shape (data_train.orig[1] * data_train.orig[2] * 3, m)
        Input features.
    Y : np.array of shape (1, m)
        True "label" vector (containing 0 if Pikachu, 1 if Rondoudou)
    layers_dims : list of length (number of layers + 1)
        Contain the input size and each layer size.
    learning_rate : float, optional
        Learning rate of the gradient descent update rule. The default is 0.0075.
    epochs : int, optional
         Number of iterations of the optimization loop. The default is 3000.
    print_cost : bool, optional
        It prints the cost every 100 steps. The default is False.

    Returns
    -------
    parameters : dict
        Parameters learnt by the model.
    costs : list
        List of costs for each epoch.

    """
    
    costs = []
    parameters = fc.initialize_parameters(layers_dims)
    
    for epoch in range(1, epochs+1):
        # Forward pass
        yhat, caches = models.model_forward(X, parameters)
        # Compute loss
        cost = fc.compute_cost(yhat, Y)
        # Backward pass
        grads = models.model_backward(yhat, Y, caches)
        # Update parameters
        parameters = models.update_parameters(parameters, grads, learning_rate)
        
        print("Training is starting...")
        if print_cost and epoch % 100 == 0 or epoch == epochs - 1:
            print("{} ------- Cost after iteration {}: {}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S'), 
                epoch, 
                np.squeeze(cost)))
        if epoch % 100 == 0 or epoch == epochs:
            costs.append(cost)
            
    return parameters, costs
        
        
### Training
parameters, costs = NN_model(data_train, 
                             label_train, 
                             layers_dims, 
                             learning_rate = 0.01,
                             epochs=1000, 
                             print_cost=True)



pred_train = models.predict(data_train, label_train, parameters)
pred_test = models.predict(data_val, label_val, parameters)


my_image = "image_test_rondou_1.jpeg"
my_label_y = [1]
path = "data/" + my_image

with Image.open(path) as im :
    im = im.resize((data_train_orig.shape[1], data_train_orig.shape[2]), resample=Image.BICUBIC)
    im = im.convert("RGB")
    im.show()
    
img_arr = np.array(im)
img_arr = img_arr / 255.

img_arr = img_arr.reshape(1, -1).T
my_predicted_image = models.predict(img_arr, my_label_y, parameters)

print ("y = " + str(np.squeeze(my_predicted_image)) +
       ", The model predicts a \"" + 
       classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +
       "\" picture !")


