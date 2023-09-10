import torch
import torch.nn as nn
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
torch.manual_seed(1)

import os
import copy

# hyperparameters
batch_size = 16



# setup - parallize model and data
def setup_parallel():
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    
    dist.init_process_group('nccl')
    torch.cuda.set_device(local_rank)
    
    return local_rank, world_size

local_rank, world_size = setup_parallel()



X = torch.rand(1024, 3).to(device)
y_transform = lambda x: (x[:, 0]**3)*7 + (x[:, 1]**2)*4 + x[:, 0]*9
y = y_transform(X).view(X.shape[0], 1)
dataset = TensorDataset(X, y)
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)


class myModel(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        
        dims = 256
        
        self.model = nn.Sequential(
            nn.Linear(input_dims, dims),
            nn.ReLU(),
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, 1),
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
    
model = myModel(X.shape[1]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# hyperparameters
epochs = 500
patience = 500
best_loss = float('inf')
best_model_weights = copy.deepcopy(model.state_dict())
no_improve = 0


for epoch in range(epochs):
    sampler.set_epoch(epoch)
    
    total_loss = 0
    ten_count = 0
    
    for X_batch, y_batch in dataloader:
    
        y_pred = model(X_batch)

        loss = loss_fn(y_pred, y_batch)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if ten_count >= 10:
            averaged_loss = total_loss / 10
            if averaged_loss < best_loss:
                best_loss = averaged_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early Stopping")
                    break
                    
        ten_count += 1

    if no_improve >= patience:
        break

    if epoch % (epochs / 10) == 0 and local_rank == 0:
        print(loss.item())
        
model.load_state_dict(best_model_weights)
        

if local_rank == 0:
    test_tensor = torch.tensor([0.1, 0.6, 0.8]).view(1, 3).to(device)    
    print(f'test = {model(test_tensor).item()}')
    print(f'true = {y_transform(test_tensor).item()}')