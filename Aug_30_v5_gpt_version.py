import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, TensorDataset
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import os
from fairscale.optim.oss import OSS
from time import time
from tqdm import trange

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.ReLU(),            
            nn.Linear(8192, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

def main():
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    # Initialize distributed environment
    dist.init_process_group(backend="nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    
    # Create the model and move it to FSDP
    model = SimpleModel()
    model = FSDP(model).cuda()

    # Create some random data
    num_samples = 1000
    X = torch.randn(num_samples, 10).cuda()
    y = (X.sum(dim=1, keepdim=True) > 0).float()  # Simple synthetic task
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    

    start_time = time()
    
    # Training loop
    for epoch in range(10):
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if local_rank == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
        
    end_time = time()
    
    total_time = end_time - start_time
    if local_rank == 0:
        print('total time = {}'.format(total_time))

if __name__ == "__main__":
    main()
