#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:29:52 2021

@author: thierryk
"""

import numpy as np
import math
import glob
from PIL import Image



def load_data(val_size=0.2):
    """
    Convert images to arrays and return them randomized through training and 
    validation set.

    Parameters
    ----------
    val_size : float, optional
        Part of validation set. The default is 0.2.

    Returns
    -------
    data_train : np.array of shape (m, HEIGHT, WIDTH, C)
        Training set.
    label_train : np.array of shape (m, 1)
        Labels of the training set.
    data_val : np.array of shape (m, HEIGHT, WIDTH, C)
        Validation set.
    label_val : np.array of shape (m, 1)
        Labels of the validation set.
    classes : np.array of shape (2,)
        Classe names : Pikachu / Rondoudou. They are encode in bytes.

    """
    list_pikachu = glob.glob('../data/pikachu/*')
    list_rondoudou = glob.glob('../data/rondoudou/*')
    
    HEIGHT = 100
    WIDTH = 100
    CHANNEL = 3
    
    classes = np.array([b'Pikachu', b'Rondoudou'])
    
    # Initialisations
    dataset_arr = np.zeros((len(list_pikachu) + len(list_rondoudou), 
                                    HEIGHT, WIDTH, CHANNEL))
    label = np.zeros((len(list_pikachu) + len(list_rondoudou), 1), dtype='int')
    
    # Generating a Pikachu array type dataset
    for k in range(len(list_pikachu)):
        with Image.open(list_pikachu[k]) as im :
            im = im.resize((HEIGHT, WIDTH), resample=Image.BICUBIC)
            im = im.convert("RGB")
        img_arr = np.array(im)
        dataset_arr[k] = img_arr
        
    # Generating a Rondoudou array type dataset
    i=0
    for k in range(len(list_pikachu), len(dataset_arr)):
        with Image.open(list_rondoudou[i]) as im2 :
            im2 = im2.resize((HEIGHT, WIDTH), resample=Image.BICUBIC)
            im2 = im2.convert("RGB")
        img_arr = np.array(im2)
        dataset_arr[k] = img_arr
        label[k] = 1
        i+=1
    
    n_samples = dataset_arr.shape[0]
    n_val = int(val_size * n_samples)
    shuffled_indices = np.random.permutation(n_samples)
    train_indices = shuffled_indices[:-n_val] 
    val_indices = shuffled_indices[-n_val:]


    data_train = dataset_arr[train_indices]
    label_train = label[train_indices]
    
    data_val = dataset_arr[val_indices]
    label_val = label[val_indices]
    
    return data_train, label_train, data_val, label_val, classes


def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Parameters
    ----------
    X : np.array
        Input data, of shape (input size, number of examples)
    Y : np.array
        True "label" vector, of shape (1, number of examples)
    mini_batch_size : int
        Size of the mini-batches
    
    Returns
    -------
    mini_batches : list
        List of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]
    mini_batches = []
    
    # Shuffling X, Y
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        start_minibatch = k * mini_batch_size
        end_minibatch = (k+1) * mini_batch_size
        mini_batch_X = shuffled_X[:, start_minibatch : end_minibatch]
        mini_batch_Y = shuffled_Y[:, start_minibatch : end_minibatch]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, end_minibatch : ]
        mini_batch_Y = shuffled_Y[:, end_minibatch : ]
    
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
    
    
    

if False :
    data_train, label_train, data_val, label_val, classes = load_data()
    
    INDEX = np.random.randint(data_train.shape[0])
    img_arr = data_train[INDEX]
    img = Image.fromarray(img_arr.astype(np.uint8))
    img.show()
    
    print ("y = " + str(label_train[INDEX]) + ". It's a " + classes[label_train[INDEX].item()].decode("utf-8") +  " picture.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    