import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import os
from time import time
start_time = time()

from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.manual_seed(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_model_parallel():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    
    return local_rank, world_size

local_rank, world_size = setup_model_parallel()

x = torch.rand((8192, 3)).to(device)
y_transform = lambda x: (x[:, 0]**3)*3 + 4*(x[:, 1]**2) + (4*x[:, 2])
# y = torch.rand(45)
y = y_transform(x).to(device)

class myModel(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.dims = 8192
        self.model = nn.Sequential(
            nn.Linear(input_dims, self.dims),
            nn.ReLU(),
            nn.Linear(self.dims, self.dims),
            nn.ReLU(),
            nn.Linear(self.dims, self.dims),
            nn.ReLU(),            
            nn.Linear(self.dims, 1)            
        )
        
    def forward(self, x):
        x = self.model(x)
        return x

    
model = myModel(x.shape[1]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

batch_size = 64
dataset = TensorDataset(x, y)
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

epochs = 50

for epoch in range(epochs):
    sampler.set_epoch(epoch)
    
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()  # Zero the gradients

        y_pred = model(x_batch)      # Get predictions from the model

        loss = loss_fn(y_pred, y_batch.view(batch_size, -1))  # Compute the loss

        loss.backward()       # Backpropagate the loss

        optimizer.step()      # Update the model parameters

    if epoch % (epochs / 10) == 0 and local_rank == 0:
        print(f'epoch = {epoch}, loss = {loss.item()}')

        
end_time = time()

total_time = end_time - start_time

if local_rank == 0:
    print(f'total_time = {total_time}')

if local_rank == 0:
    print('test')
    print(model(torch.tensor([0.3,0.4,0.5]).to(device)))
    print('real')
    print(y_transform(torch.tensor([[0.3,0.4,0.5]]).to(device)))    