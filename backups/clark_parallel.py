import torch
import torch.nn as nn
import torch.distributed as dist
from fairscale.nn.model_parallel import ColumnParallelLinear, RowParallelLinear
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import time
import os
from typing import Tuple

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = ColumnParallelLinear(10, 8000)
        self.fc2 = RowParallelLinear(8000, 16000)
        self.fc3 = ColumnParallelLinear(16000, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    dist.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    return local_rank, world_size

def main():
    # Setup model parallel
    local_rank, world_size = setup_model_parallel()

    # Create a model
    model = Net().cuda(local_rank)

    # Create a MSE loss function
    criterion = nn.MSELoss()

    # Create an optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create some fake data
    x = torch.randn((32, 10)).cuda(local_rank)  # 32 samples, 10 features
    y = torch.randn(32, 1).cuda(local_rank)     # 32 samples, 1 target variable

    #start time
    start_time = time.time()
    
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
            
    end_time = time.time()
    
    total_time = end_time - start_time
    
    print(total_time)

if __name__ == "__main__":
    main()
