# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 23:15:37 2017

@author: USER
"""

def save_matlab_data(data, filename):
    with open(filename, 'w') as f:
        for i in data:
           f.write("%d," % i)