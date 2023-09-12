import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.optim.oss import OSS

from time import time
start_time = time()

import os
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyperparameters
batch_size = 16


# setup model and data parallel
def setup_parallel():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    
    dist.init_process_group('nccl')
    torch.cuda.set_device(local_rank)
    
    return local_rank, world_size

local_rank, world_size = setup_parallel()
    
    


X = torch.rand(1024, 3).to(device)
y_transform = lambda x: (x[:,0]**3)*7 + (x[:,1]**2)*2 + x[:,2]*4
y = y_transform(X).view(X.shape[0], 1)
dataset = TensorDataset(X, y)
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

class myModel(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        
        dims = 4096
        
        self.model = nn.Sequential(
            nn.Linear(input_dims, dims),
            nn.ReLU(),
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, 1)
        )

    def forward(self, x):
        
        x = self.model(x)
        return x
    

model = myModel(X.shape[1]).to(device)
model = FSDP(model)
optimizer = OSS(params=model.parameters(), optim=optim.AdamW, lr=0.001)
loss_fn = nn.MSELoss()

epochs = 10

for epoch in range(epochs):
    sampler.set_epoch(epoch)
    
    for X_batch, y_batch in dataloader:
    
        y_pred = model(X_batch)

        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % (epochs / 10) == 0 and local_rank == 0:
        print(loss.item())
        
model.eval()
with torch.no_grad():
    
    dummy_batch = torch.rand(batch_size, 3).to(device)

    test_tensor = torch.tensor([0.3, 0.7, 0.1]).view(1, 3).to(device)    
    
    dummy_batch[0] = test_tensor
    
    output = model(dummy_batch)[0].item()
    
    if local_rank == 0:
        
        print(f'test = {output}')
        print(f'true = {y_transform(test_tensor).item()}')

if local_rank == 0:
    end_time = time()
    total_time = end_time - start_time
    print(f'time = {total_time}')