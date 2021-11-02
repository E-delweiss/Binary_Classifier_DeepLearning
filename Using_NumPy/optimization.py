#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:32:03 2021

@author: thierry
"""

import numpy as np
import math

def optimization(X, y, epoch, learning_rate, minibatch_size, algorithm, 
                 beta = 0.9, beta1 = 0.9, beta2 = 0.999, sigma = 10e-8, decay_rate=0, timeInterval=1):
    
    m = X.shape[1]
    
    
    
    learning_rate = 1 / (1 + decay_rate * math.floor(epoch / timeInterval))
    
    if algorithm == "SGD"