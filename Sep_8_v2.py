import torch
import torch.nn as nn
import torch.distributed as dist

import torch.optim as optim

import os

from tqdm import trange

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

torch.manual_seed(1)

# hyperparameters
batch_size = 16

def setup_parallel():
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    
    return local_rank, world_size

local_rank, world_size = setup_parallel()

X = torch.rand((4096, 3)).to(device)
y_transform = lambda x: (x[:, 0]**3) + (x[:, 1]**2) + x[:, 2]
y = y_transform(X)
dataset = TensorDataset(X, y.view(X.shape[0], 1))
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

class myModel(nn.Module):
    
    def __init__(self, input_dims):
        super().__init__()
        
        self.dims = 256
        self.model = nn.Sequential(
            nn.Linear(input_dims, self.dims),
            nn.ReLU(),
            nn.Linear(self.dims, self.dims),
            nn.ReLU(),
            nn.Linear(self.dims, 1)
        )
        
    def forward(self, x):
        x = self.model(x)
        
        return x
    

model = myModel(X.shape[1]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()


epochs = 100

for epoch in range(epochs):
    
    for X_batch, y_batch in dataloader:
        
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = loss_fn(y_pred, y_batch)
        
        loss.backward()
        
        optimizer.step()
        
    if epoch % (epochs / 10) == 0 and local_rank == 0:
        print(loss.item())


if local_rank == 0:
    print('pred')
    print(model(torch.tensor([0.2, 0.3, 0.7]).to(device)).item())
    print('true')
    print(y_transform(torch.tensor([[0.2, 0.3, 0.7]])).item())