import math
import warnings

from torch import Tensor
import torch
import numpy as np

perm_matrix = torch.nn.Linear(40,40, bias=False)

delta=19  
p_array=torch.zeros(40, dtype=torch.long)
for i in range(40):
    p_array[i]=(i*delta)%(40)


matrx = torch.zeros((40,40))

#print(perm_matrix.weight.data)

counter = 0
for i in p_array:
    matrx[i, counter] = 1
    counter = counter + 1


k = 1/40
perm_matrix.weight.data = matrx + torch.FloatTensor(40, 40).uniform_(-np.sqrt(k), np.sqrt(k))

print(perm_matrix.weight.data)
# print(matrx)
# print(p_array)

inputs = torch.randint(0, 2, (10, 40, 1), dtype=torch.float)

##Regular
inputs1 = inputs.permute(1,0,2)
res    = inputs1[p_array]
res    = res.permute(1, 0, 2)



# ##Matrix form
inputs2 = torch.transpose(inputs, 1,2)
res2    = perm_matrix(inputs2)
res2 = torch.transpose(res2, 1,2)

print(res2[0,:,:] - res[0,:,:])