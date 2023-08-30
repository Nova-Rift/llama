# Aug_27_v2.py

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def train(rank, world_size):
    # Set up the device and allow the environment to decide the specific GPU
    device = torch.device(f"cuda:{rank}")

    # Initialize the distributed environment
    dist.init_process_group(
        'nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

    # Create model and move to appropriate device
    model = SimpleModel().to(device)
    fsdp_model = FSDP(model)

    # Ensure criterion is also on the correct device
    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD(fsdp_model.parameters(), lr=0.01)

    inputs = torch.randn(20, 10).to(device)
    targets = torch.randn(20, 10).to(device)

    for epoch in range(1000):  # loop over the dataset multiple times
        optimizer.zero_grad()
        outputs = fsdp_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if rank == 0 and epoch % 100 == 0:  # Only print from one process to avoid clutter
            print('Epoch {}: Loss: {:.4f}'.format(epoch, loss.item()))

    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    train(rank, world_size)
