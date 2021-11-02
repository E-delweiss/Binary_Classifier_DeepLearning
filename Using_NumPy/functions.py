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
    cache : np.array 
        Storing variable for backward pass

    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    
    return A, cache


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
    s = 1/(1+np.exp(-Z))
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
    cache = Z
    return A, cache


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
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
    
    return parameters


def linear_forward(A, W, b):
    """
    Forward pass before activation

    Parameters
    ----------
    A : np.array of shape (n[l-1], m)
        Activations from previous layer (or input data).
    W : np.array of shape (n[l], n[l-1])
        Weights matrix.
    b : np.array of size (n[l], 1)
        Bias vector.

    Returns
    -------
    Z : np.array of size (layer_dims[l],1)
        Pre-activation parameter.
    cache : tuple
        Storing variable for computing the backward pass efficiently.

    """
    Z = np.dot(W,A) + b
    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward pass LINEAR->ACTIVATION layer

    Parameters
    ----------
    A_prev : np.array(n[l-1], m)
        Activations from previous layer (or input data).
    W : np.array(n[l], n[l-1])
        Weights matrix of the current layer.
    b : np.array(n[l], 1)
        Bias vector of the current layer..
    activation : str
        Name of the activation function.

    Returns
    -------
    A : np.array
        Post-activation value.
    cache : tuple
        containing "linear_cache" and "activation_cache". Storing variable for 
        computing the backward pass efficiently

    """
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b) 
        A, activation_cache = sigmoid(Z)
        
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b) 
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    
    return A, cache


def compute_cost(AL, Y):
    """
    Compute the cost function, sum of the loss function

    Parameters
    ----------
    AL : np.array of shape (1, m)
        Probability vector corresponding to the label predictions. Also called yhat.
    Y : np.array of shape (1, m)
        True "label" vector.

    Returns
    -------
    cost : float
        Cross-entropy cost.

    """
    
    m = Y.shape[1]
    cost = (-1/m) * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
    cost = np.squeeze(cost)
    
    return cost


def compute_cost_L2regularization(AL, Y, layers_dim, parameters, lambd):
    """
    Compute the cost function, sum of the loss function, with L2 regularization.

    Parameters
    ----------
    AL : np.array of shape (1, m)
        Probability vector corresponding to the label predictions. Also called yhat.
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
    m = AL.shape[1]
    cross_entropy_cost = compute_cost(AL, Y)
    cst = (1/m)*(lambd/2)
    
    somme = 0
    for l in range(1, len(layers_dim)):
        W = parameters["W"+str(l)]
        somme = somme + np.sum(np.square(W))
    
    L2_regularization_cost = cst * somme
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

def linear_backward(dZ, cache):
    """
    Compute the backward pass

    Parameters
    ----------
    dZ : np.array of shape (n[l],1)
        Gradient of the cost function with respect to the linear output 
        (of current layer l).
    cache : tuple
        Containing (A_prev, W, b) coming from the forward propagation in the 
        current layer.

    Returns
    -------
    dA_prev : np.array of shape A.shape
        Gradient of the cost with respect to the activation (previous layer l-1)
    dW : np.array of shape W.shape
        Gradient of the cost with respect to the activation (current layer)
    db : np.array of shape b.shape
        Gradient of the cost with respect to b (current layer l).

    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = (1./m) * np.dot(dZ, A_prev.T)
    db = (1./m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev =  np.dot(W.T, dZ)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, dropout=None):
    """
    Implement the backward pass for the LINEAR->ACTIVATION layer.

    Parameters
    ----------
    dA : np.array
         Post-activation gradient for current layer l.
    cache : tuple
        Storing variable for computing backward propagation efficiently.
    activation : str
        Activation to be used in this layer, stored as a text string: "sigmoid" or "relu".

    Returns
    -------
    dA_prev : np.arry of shape dA.shape
        Post-activation gradient for current layer l-1.
    dW : np.array 
        Gradient of the cost with respect to W (current layer l).
    db : TYPE
        Gradient of the cost with respect to b (current layer l).

    """
    linear_cache, activation_cache = cache
    
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db