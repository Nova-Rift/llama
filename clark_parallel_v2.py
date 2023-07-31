import os

from tqdm import trange

import torch
import torch.nn as nn
import torch.distributed as dist

from fairscale.nn.model_parallel import ColumnParallelLinear, RowParallelLinear
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from typing import Tuple

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = ColumnParallelLinear(10, 8000)
        self.fc2 = ColumnParallelLinear(8000, 16000)
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
    
    model = Model().cuda(local_rank)
    
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    x = torch.randn((32, 10)).cuda(local_rank)
    y = torch.randn(32, 1).cuda(local_rank)
    
    for epoch in range(100):
        
        outputs = model(x)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print('Epoch {} / {}, Loss: {}'.format(epoch+1, 100, loss.item()))
                  
if __name__ == "__main__":
    main()