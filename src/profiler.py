''' 
Pytorch model profiler
'''

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from inference import *
from network import *


model=load_model('path')

input_shape=torch.rand(1,1,224,224)

with profile(activities=[ProfilerActivity.CPU], profile_memory=True,record_shapes=True) as prof:
    with record_function("inference"):
        model(input_shape)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


