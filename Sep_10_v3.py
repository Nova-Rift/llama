import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.distributed as dist
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

import os, copy
import warnings
warnings.filterwarnings("ignore")



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



X = torch.rand(1024 , 3).to(device)
y_transform = lambda x : (x[:, 0]**3)*3 + (x[:, 1]**2) + x[:, 2]
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
params = sum(p.numel() for p in model.parameters())/1e6
if local_rank == 0:
    print(f'model parameters = {params} million')
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

epochs = 5

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
         


# if local_rank == 0:

# Switch the model to evaluation mode
model.eval()

with torch.no_grad():
    if local_rank == 0:
        print('epochs done')

    # Create a dummy batch with size 16
    dummy_tensor = torch.zeros(batch_size, X.shape[1]).to(device)
    # Set the first tensor to our test tensor
    dummy_tensor[0] = torch.tensor([0.3, 0.7, 0.4]).to(device)

    # Run inference
    outputs = model(dummy_tensor)
    if local_rank == 0:    
        print(f'test = {outputs[0].item()}')  # Only consider the output from our test tensor
        print(f'true = {y_transform(dummy_tensor[0].view(1, 3)).item()}')

# Switch the model back to training mode for future training
model.train()

