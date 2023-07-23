import os
import torch
import torch.nn as nn
import torch.distributed as dist
from fairscale.nn.model_parallel import ColumnParallelLinear, RowParallelLinear
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = ColumnParallelLinear(10, 8000)
        self.fc2 = ColumnParallelLinear(8000, 16000)
        self.fc3 = RowParallelLinear(16000, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    # Get the number of GPUs available
    world_size = torch.cuda.device_count()

    # Initialize the distributed environment
    dist.init_process_group(backend='nccl')

    # Initialize model parallel utils
    initialize_model_parallel(world_size)

    # Create a model
    model = Net().cuda()

    # Create a MSE loss function
    criterion = nn.MSELoss()

    # Create an optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create some fake data
    x = torch.randn((32, 10)).cuda()  # 32 samples, 10 features
    y = torch.randn(32, 1).cuda()     # 32 samples, 1 target variable

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
