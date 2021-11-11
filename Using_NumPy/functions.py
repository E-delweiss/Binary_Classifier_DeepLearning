#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 18:12:43 2021

@author: thierryk
"""

import numpy as np


def sigmoid(Z):
    """
    Calculate the sigmoid activation function

    Parameters
    ----------
    Z : np.array of any shape
        pre-activation parameter.
        
    Returns
    -------
    A : np.array of shape Z.shape
        Post-activation parameter

    """
    A = 1 / (1 + np.exp(-Z))
    return A


def sigmoid_backward(dA, cache):
    """
    Implements the backward propagation for SIGMOID unit

    Parameters
    ----------
    dA : np.array of any shape
        Post-activation gradient
    cache : TYPE
        'Z' used to compute backward propagation efficiently.

    Returns
    -------
    dZ : np.array of shape dA.shape
        Gradient of the cost with respect to Z.

    """
    Z = cache
    s = sigmoid(Z)
    dZ = dA * s * (1 - s)
    
    return dZ


def relu(Z):
    """
    Calculate the ReLU activation function

    Parameters
    ----------
    Z : np.array of any shape
        pre-activation parameter.
        
    Returns
    -------
    A : np.array of shape Z.shape
        Post-activation parameter
    cache : np.array
        Storing variable for backward pass

    """
    A = np.maximum(0,Z)    
    
    return A


def relu_backward(dA, cache):
    """
    Implements the backward propagation for ReLU unit
    
    Parameters
    ----------
    dA : np.array of any shape
        Post-activation gradient
    cache : TYPE
        'Z' used to compute backward propagation efficiently.

    Returns
    -------
    dZ : np.array of shape dA.shape
        Gradient of the cost with respect to Z.

    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    return dZ


def initialize_parameters(layer_dims):
    """
    Initialize randomly parameters W and b of each layer

    Parameters
    ----------
    layer_dims : list[np.array]
         Containing the dimensions of each layer in the network

    Returns
    -------
    parameters : dict
        Contain parameters "W1", "b1", ..., "WL", "bL":

    """
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
    
    return parameters


def forward_pass(A_prev, W, b, activation):
    """
    Implement the forward pass
    LINEAR->ACTIVATION layer

    Parameters
    ----------
    A : np.array of shape (n[l-1], m)
        Activations from previous layer (or input data).
    W : np.array of shape (n[l], n[l-1])
        Weights matrix.
    b : np.array of size (n[l], 1)
        Bias vector.
    activation : str
        Name of activation function.

    Returns
    -------
    A : np.array
        Post-activation value.
    cache : tuple
        containing "linear_cache" and "activation_cache". Storing variable for 
        computing the backward pass efficiently.

    """
    # print("Shape of W*Aprev : ", np.dot(W, A_prev).shape)
    # print("Shape of b : ", b.shape)
    Z = np.dot(W, A_prev) + b
    
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)

    linear_cache = (A_prev, W, b)
    activation_cache = Z
    cache = (linear_cache, activation_cache)
    
    return A, cache


def compute_cost(yhat, Y, mini_batch_size=None):
    """
    Compute the cost function, sum of the loss function

    Parameters
    ----------
    yhat : np.array of shape (1, m)
        Probability vector corresponding to the label predictions. It's actually
        the activation at the layer L : AL.
    Y : np.array of shape (1, m)
        True "label" vector.
    mini_batch_size : int, optionnal
        Arg used to trigger the normalization of the cost by 1/m if there is no
        mini_batches. Default is None.

    Returns
    -------
    cost : float
        Cross-entropy cost function with or without dividing by number of 
        training examples

    """
    AL = yhat

    cost = np.sum(-np.log(AL)*Y - np.log(1-AL)*(1-Y))  
    if mini_batch_size is None:
        m = Y.shape[1]
        cost = (1/m) * cost

    return cost


def compute_cost_L2regularization(yhat, Y, layers_dim, parameters, lambd, mini_batch_size=None):
    """
    Compute the cost function, sum of the loss function, with L2 regularization.

    Parameters
    ----------
    yhat : np.array of shape (1, m)
        Probability vector corresponding to the label predictions. It's actually
        the activation at the layer L : AL.
    Y : np.array of shape (1, m)
        True "label" vector.
    parameters : dict
        Output of initialize_parameters_deep().
    lambd : float
        Regularization factor

    Returns
    -------
    cost : float
        Cross-entropy cost.

    """
    AL = yhat
    m = AL.shape[1]
    cross_entropy_cost = compute_cost(AL, Y, mini_batch_size)
    
    somme = 0
    for l in range(1, len(layers_dim)):
        W = parameters["W"+str(l)]
        somme = somme + np.sum(np.square(W))
    
    L2_regularization_cost = (1/m)*(lambd/2) * somme
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost



def backward_pass(dA, cache, activation):
    """
    Compute the backward pass :
    LINEAR -> dZ -> GRADS

    Parameters
    ----------
    dA : np.array
         Post-activation gradient for current layer l.
    cache : tuple
        containing "linear_cache" and "activation_cache". These caches come from 
        the forward pass.
    activation : str
        Activation to be used in this layer, stored as a text string: "sigmoid" or "relu".

    Returns
    -------
    dA_prev : np.array of shape A.shape
        Gradient of the cost with respect to the activation (previous layer l-1)
    dW : np.array of shape W.shape
        Gradient of the cost with respect to the activation (current layer)
    db : np.array of shape b.shape
        Gradient of the cost with respect to b (current layer l).

    """
    linear_cache, activation_cache = cache
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    
    ### Compute grads
    dW = (1./m) * np.dot(dZ, A_prev.T)
    db = (1./m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev =  np.dot(W.T, dZ)

    return dA_prev, dW, db










