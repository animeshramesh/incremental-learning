import numpy as np
from collections import defaultdict

'''
Returns a (32, 32, 3) image for a given array of 3072 pixels
'''
def reshape_img(pixels):
    r_channel = pixels[0:1024].reshape((32,32))
    g_channel = pixels[1024:2048].reshape((32,32))
    b_channel = pixels[1024:2048].reshape((32,32))
    return np.dstack((r_channel, g_channel, b_channel))

'''
Returns a custom dataset specific for incremental learning. 
The CIFAR dataset is broken down into 4 sub-datasets which would be used for incremental learning.
    - training : 40 classes which would be used for training our img classification network.
    - base : 20 classes which are considered as 'base'.
    - simulation : 20 classes. The new class would be picked from this category
    - test : 20 classes which would be used for testing our transformation network.
    
All the sub-datasets are mutually exclusive. 
'''
def create_custom_dataset(cifar):
    dataset = {'training': defaultdict(list), 'base':defaultdict(list) , 'simulation':defaultdict(list), 'test':defaultdict(list)}
    for ind in range(len(cifar['data'])):
        j = cifar['fine_labels'][ind]
        if j < 40:
            dataset['training'][j].append(reshape_img(cifar['data'][ind]))
        elif j >= 40 and j < 60:
            dataset['base'][j].append(reshape_img(cifar['data'][ind]))
        elif j >= 60 and j < 80:
            dataset['simulation'][j].append(reshape_img(cifar['data'][ind]))
        else:
            dataset['test'][j].append(reshape_img(cifar['data'][ind]))
    return dataset 
