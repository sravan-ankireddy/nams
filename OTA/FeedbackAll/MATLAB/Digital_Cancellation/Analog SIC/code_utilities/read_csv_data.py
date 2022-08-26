# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:25:02 2017

@author: USER
"""


import numpy as np
import pandas as pd
import sys

print ('format : read filename, column name 1, column name 2')
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

# save 1-d data
def save_data(data, filename):
    with open(filename, 'w') as f:
        for i in data:
           f.write("%d\n" % i)

# save iq data
def save_iq(data_i,data_q, filename):
    with open(filename, 'w') as f:
        for i in range(1,len(data_i)):
           f.write("%d %d\n" % (data_i[i],data_q[i]))

rd_filename = sys.argv[1]    
data_all = pd.read_csv(rd_filename)

save_filename = 'temp.txt'
  
if len(sys.argv) == 3:
   col_name1 = sys.argv[2];
   data = data_all[col_name1];  
   save_data(data, save_filename)
   
elif len(sys.argv) == 4: 
   col_name1 = sys.argv[2];
   col_name2 = sys.argv[3];
   
   data_i = data_all[col_name1];                                         
   data_q = data_all[col_name2];        
                            
   save_iq(data_i,data_q,save_filename)




