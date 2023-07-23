import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel

import matplotlib.pyplot as plt
plt.style.use('ggplot')

class myModel(nn.Module):
    def __init__(self, input_dims):
        super().__init__()

        self.layer1 = nn.Linear(input_dims, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        output = self.layer3(x)

        return output

def main():
    # Initialize the distributed environment.
    dist.init_process_group('nccl')
    
    # Here we simulate a input tensor of [1,3,4]
    X = torch.tensor([1,3,4]).float().cuda()
    X.shape[0]

    # Create the model and move it to GPU
    model = myModel(X.shape[0]).cuda()
    model = DistributedDataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Train the model
    model(X)

    # Save
    torch.save({
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'checkpoint.pth')

if __name__ == "__main__":
    main()
