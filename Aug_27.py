# simple_ddp.py

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

# Ensure you have set CUDA_VISIBLE_DEVICES properly before running this script.

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def train(rank, world_size):
    # Set up the device and move the model to that device
    device = torch.device("cuda:{}".format(rank))
    model = SimpleModel().to(device)

    # Initialize the distributed environment
    dist.init_process_group(
        'nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    inputs = torch.randn(20, 10).to(device)
    targets = torch.randn(20, 10).to(device)

    for epoch in range(1000):  # loop over the dataset multiple times
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
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
