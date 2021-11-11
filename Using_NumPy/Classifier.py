#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 08:22:13 2021

@author: thierryk
"""

import numpy as np
import datetime
from PIL import Image

from import_data import load_data, random_mini_batches
import functions as fc
import models
import optim

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


def NN_model(X, Y, layers_dims, optimizer='gd', learning_rate = 0.0075, mini_batch_size=64, lambd=0, keep_prob=1, beta=0.9,
             beta1=0.9, beta2= 0.999, epsilon=1e-8, epochs = 1000):
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
    t = 0
    m = X.shape[1]
    
    # Initialize parameters
    parameters = fc.initialize_parameters(layers_dims)
    
    # Initialize parameters optimizer
    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = optim.initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = optim.initialize_adam(parameters)
    
    
    print(f"Training is starting with minibatch size = {mini_batch_size} ...")
    for epoch in range(1, epochs+1):
        # Chose a minibatch per epoch
        minibatches = random_mini_batches(X, Y, mini_batch_size)
        cost_total = 0
        
        # Training through minibatch
        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch 
        
            # Forward pass
            yhat, caches, listD = models.forward(minibatch_X, parameters, keep_prob)
            
            # Compute cost and add to the cost total
            if lambd == 0:
                cost = fc.compute_cost(yhat, minibatch_Y, mini_batch_size)
            else:
                cost = fc.compute_cost_L2regularization(yhat ,minibatch_Y, layers_dims, parameters, lambd, mini_batch_size)
            cost_total += cost
        
            # Backward pass
            grads = models.backward(yhat, minibatch_Y, caches, lambd, listD, keep_prob)
        
            # Update parameters
            if optimizer == "gd":
                parameters = models.update_parameters(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = optim.momentum_update_parameters(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                parameters, v, s, _, _ = optim.adam_update_parameters(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)
        # Compute average cost
        cost_avg = cost_total / m
            
        # Printing
        if epoch==1 or epoch % 100 == 0 or epoch == epochs: #- 1:
            print("{} ------- Cost after iteration {}: {}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S'), 
                epoch, 
                cost_avg))
        if epoch % 100 == 0 or epoch == epochs:
            costs.append(cost_avg)
            
    return parameters, costs
        
        
### Training
parameters, costs = NN_model(data_train, 
                             label_train, 
                             layers_dims, 
                             learning_rate = 0.01,
                             lambd=0,
                             keep_prob=1,
                             beta = 0.9,
                             beta1 = 0.9,
                             beta2 = 0.999,
                             epsilon=1e-8,
                             epochs = 400)


pred_train = models.predict(data_train, label_train, parameters)
pred_test = models.predict(data_val, label_val, parameters, training_set=False)





test = False
if test :
    my_image = "image_test_rondou_1.jpeg"
    my_label_y = [1]
    path = "../data/" + my_image
    
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


