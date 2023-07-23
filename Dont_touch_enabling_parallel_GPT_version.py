import os
import torch
import torch.nn as nn
import torch.distributed as dist
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDP
from fairscale.optim import OSS

device = 'cuda' if torch.cuda.is_available else 'cpu'

# Define a simple three-layered neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # input dimension is 10, output dimension is 20
        self.fc2 = nn.Linear(20, 30)  # input dimension is 20, output dimension is 30
        self.fc3 = nn.Linear(30, 1)   # input dimension is 30, output dimension is 1

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize distributed communication
dist.init_process_group(backend='nccl')

# Create a model and move it to GPU
model = Net().to(device)

# Create a MSE loss function
criterion = nn.MSELoss()

# Create a distributed optimizer (Adam) using FairScale's Optimizer State Sharding
optimizer = OSS(model.parameters(), optim=torch.optim.Adam, lr=0.1)

# Wrap the model using FairScale's Sharded Data Parallel
model = ShardedDP(model, optimizer)

# Create some fake data
x = torch.randn((32, 10)).to(device)  # 32 samples, 10 features
y = torch.randn(32, 1).to(device)     # 32 samples, 1 target variable

# Training loop for 100 epochs
for epoch in range(100):
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{100}, Loss: {loss.item()}')
