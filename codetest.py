import numpy
import torch
a=torch.tensor([-0.8858,0.3241,0.9456])
active_func=torch.nn.Sigmoid()
a_sigmoid=active_func(a)
label=torch.FloatTensor([0,1,1])
loss=torch.nn.BCELoss()
loss_bce=loss(a_sigmoid,label)
print(a)
print(a_sigmoid)
print(label)
print(loss_bce)