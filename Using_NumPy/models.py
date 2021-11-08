#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 21:46:32 2021

@author: thierryk
"""

import numpy as np

import functions as fc

def model_forward(X, parameters, keep_prob=1):
    """
    Implement the linear portion of forward propagation for a single layer (layer l)

    Parameters
    ----------
    X : np.array of shape (n[0], m)
        Data input.
    parameters : dict
        Output of initialize_parameters_deep().
    keep_prob : int, optional
        Dropout factor : trigger the dropout regularization if < 1. The default is 1.

    Returns
    -------
    AL : np.array
        Activation value from the output (last) layer, i.e. prediction vector,
        also called yhat
    caches : list
        caches containing: every cache of linear_activation_forward() 
        (there are L of them, indexed from 1 to L).
    listD : list
        list containing every mask used to dropout the layer [l]. 
        (there are L-1 of them, indexed from 1 to L-1).

    """
    caches = []
    A = X
    L = len(parameters) // 2
    listD = []
    
    # for l in range(1, L+1):
    #     print("a la couche " + str(l) + " : " + str(parameters["W" + str(l)].shape))
    
    ### Hidden Layers
    for l in range(1, L):
        A_prev = A
        W = parameters["W"+str(l)]
        b = parameters["b"+str(l)]
        A, cache = fc.forward_pass(A_prev, W, b, activation="relu")
        caches.append(cache)
        
        # Handle dropout
        if keep_prob < 1:
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D<keep_prob).astype(int) # mask creation
            A = A * D
            A = A / keep_prob
            listD.append(D)
    
    ### Output layer
    A_prevL = A
    WL = parameters["W"+str(l+1)]
    bL = parameters["b"+str(l+1)]
    AL, cache = fc.forward_pass(A_prevL, WL, bL, activation = "sigmoid")
    caches.append(cache)
    
    return AL, caches, listD



def model_backward(AL, Y, caches, lambd=0, listD=[], keep_prob=1):
    """
    Implement the backward propagation for the 
    [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group.
    Can use the L2 and/or dropout regularizations.

    Parameters
    ----------
    AL : np.array of shape Y.shape
        Probability vector, output of the forward propagation (L_model_forward()).
        Also called yhat.
    Y : np.array
        True "label" vector (containing 0 or 1).
    caches : list
        Contain: every cache of linear_activation_forward() with "relu" 
        (it's caches[l], for l in range(L-1) i.e l = 0...L-2) and the cache 
         of linear_activation_forward() with "sigmoid" (it's caches[L-1]).
    lambd : float
        L2 regularization factor : trigger the L2reg if > 0. The default is 0.
    listD : list
        list containing every mask used to dropout the layer [l]. 
        (there are L-1 of them, indexed from 1 to L-1).
    keep_prob : int, optional
        Dropout factor : trigger the dropout regularization if < 1. The default is 1.
    
    Returns
    -------
    grads : dict
        A dictionary with the gradients.

    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    ### Output layer L (using dAL to get dAprev i.e. dA[L-1])
    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = fc.backward_pass(dAL, current_cache, 'sigmoid')
    
    ### L2 regularization terms initialization
    L2reg_dW = np.array([0.])
    L2reg_db = np.array([0.])
    
    ### Handle L2 regularization for output layer
    if lambd > 0 :
        L2reg_dW = (lambd/m) * dW_temp
        L2reg_db = (lambd/m) * db_temp
    
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp + L2reg_dW
    grads["db" + str(L)] = db_temp + L2reg_db
    
    ### Hidden layers : from layer L-1 to layer 1 using dA[L-1] to get dA[L-2]... until dA[0]
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = fc.backward_pass(grads["dA" + str(l+1)], current_cache, 'relu')
        
        # Handle dropout 
        if keep_prob < 1:
            print(len(listD))
            dA_prev_temp = dA_prev_temp * listD[l]
            dA_prev_temp = dA_prev_temp / keep_prob
        
        # Handle L2 regularization for hidden layers
        if lambd > 0:
            L2reg_dW = (lambd/m) * dW_temp
            L2reg_db = (lambd/m) * db_temp
            
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp + L2reg_dW
        grads["db" + str(l + 1)] = db_temp + L2reg_db
        
    return grads



def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Parameters
    ----------
    params : dict
        Containing the parameters.
    grads : dict
        Contain the grads output of model_backward.
    learning_rate : float
        Tune the rate of learning.

    Returns
    -------
    parameters : dict
        Contain the update parameters.

    """
    # parameters = params.copy()
    L = len(parameters) // 2
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - \
            learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - \
            learning_rate * grads["db" + str(l+1)]
    

    return parameters


def predict(X, y, parameters, training_set=True):
    """
    Function used to predict the results of a  L-layer neural network.

    Parameters
    ----------
    X : np.array
        Input dataset.
    y : np.array
        True "label" vector (containing 0 or 1).
    parameters : dict
        Parameters of the trained model.
    training_set : bool, optional
        Verbose parameter

    Returns
    -------
    p : TYPE
        DESCRIPTION.

    """
    if training_set : 
        name = "Training"
    else:
        name = "Validation"
    
    m = X.shape[1]
    p = np.zeros((1,m))
    
    # Forward pass
    probas, caches, _ = model_forward(X, parameters)
    
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
                  
    print(f"{name} Accuracy: " + str(np.sum((p == y)/m)))
    
    return p











