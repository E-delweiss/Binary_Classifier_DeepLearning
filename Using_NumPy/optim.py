# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 09:49:07 2021

@author: lt54099
"""

import numpy as np

def initialize_velocity(parameters):
    """
    Initializes the velocity for the momentum optimization as a python 
    dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the 
                corresponding gradients/parameters.
    Parameters
    ----------
    parameters : dict
        Contain the weights and bias at the layer l
    
    Returns
    -------
    v : dict
        Contain the current velocity at the layer l for the corresponding gradients.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    L = len(parameters) // 2
    v = {}
    
    # Initialize velocity
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)        
    return v



def momentum_update_parameters(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using gradient descent

    Parameters
    ----------
    parameters : dict
        Containing the parameters.
    grads : dict
        Contain the grads output of the backward pass.
    v : dict
        Contain the current velocity at the layer l for the corresponding gradients.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    beta : float
        Momentum hyperparameter
    learning_rate : float
        Tune the rate of learning.

    Returns
    -------
    parameters : dict
        Contain the update parameters.
    v : dict
        Contain the current velocity at the layer l for the corresponding gradients.
    
    """
    L = len(parameters) // 2
    
    # Momentum update for each parameter
    for l in range(1, L+1):
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)] 
        
        parameters["W" + str(l)] = parameters["W" + str(l)] - \
            learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - \
            learning_rate * v["dW" + str(l)]
    
    return parameters, v


def initialize_adam(parameters):
    """
    Update parameters using Adam
    
    Parameters
    ----------
    parameters : dict
         Initializes v and s as two python dictionaries.
        
    Returns
    -------
    v : dict
        Will contain the exponentially weighted average of the gradient. Initialized with zeros.
    s : dict
        Will contain the exponentially weighted average of the squared gradient. Initialized with zeros.

    """
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        
    return v, s



def adam_update_parameters(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam
    
    Parameters
    ----------
    parameters : dict
        Containing the parameters for the current layer l
    v : dict
        Will contain the exponentially weighted average of the gradient.
    s : dict
        Will contain the exponentially weighted average of the squared gradient.
    t : int
        Adam variable, counts the number of taken steps
    learning_rate : float
        Learning_rate
    beta1 : float
        Exponential decay hyperparameter for the first moment estimates 
    beta2 : float
        Exponential decay hyperparameter for the second moment estimates 
    epsilon : float
        Hyperparameter preventing division by zero in Adam updates
        
    Returns
    -------
    parameters : dict
        Containing the parameters for the current layer l
    v : dict
        Will contain the exponentially weighted average of the gradient. Initialized with zeros.
    s : dict
        Will contain the exponentially weighted average of the squared gradient. Initialized with zeros.
    """ 
    L = len(parameters) // 2
    v_corrected = {}                         
    s_corrected = {}
    
    for l in range(1, L+1):
        # Exponentially moving average of the gradients
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
        
        # Compute bias-corrected for first moment estimate
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - pow(beta1, t))
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - pow(beta1, t))
        
        # Exponentially moving average of the squared gradients
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * pow(grads["dW" + str(l)], 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * pow(grads["db" + str(l)], 2)
        
        # Compute bias-corrected for second moment estimate
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - pow(beta2, t))
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - pow(beta2, t))
        
        # Update parameters
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)
        
        return parameters, v, s, v_corrected, s_corrected
        
        
        
        
        
        
        
        
        






