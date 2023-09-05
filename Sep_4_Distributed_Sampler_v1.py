import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
import torch.distributed as dist

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def setup_model_parallel():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    return local_rank, world_size

local_rank, world_size = setup_model_parallel()

# Generate structured data
input_size = 10
num_samples = 100
X = torch.rand(num_samples, input_size)
y = X.sum(dim=1, keepdim=True)

# Create DataLoader with DistributedSampler
batch_size = 16
dataset = TensorDataset(X, y)
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = FSDP(SimpleNN(input_size).cuda())

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Train the model
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    sampler.set_epoch(epoch)

    for i, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(inputs.cuda())

        loss = criterion(outputs, targets.cuda())

        loss.backward()

        optimizer.step()

    if local_rank == 0 and (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

if local_rank == 0:
    print('Training complete')

dist.destroy_process_group()
