import torch
import torch.distributed as dist
from torch.nn import Linear, MSELoss
from torch.optim import SGD
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.optim.oss import OSS
import torch.optim as optim

import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)



from fairscale.nn.model_parallel.initialize import initialize_model_parallel

def setup_model_parallel():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

local_rank, world_size = setup_model_parallel()

# Create a simple model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = Linear(5, 8192)
        self.layer2 = Linear(8192, 8192)
        self.layer3 = Linear(8192, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)

# Wrap the model with FSDP
# model = FSDP(SimpleModel()).cuda()
device = 'cuda'
model = FSDP(SimpleModel().to(device))
optimizer = OSS(params=model.parameters(), optim=optim.AdamW, lr=0.001)



# Sample data
data = torch.randn(20, 5).cuda()
target = torch.randn(20, 1).cuda()
loss_func = MSELoss()

# Training loop
for epoch in range(200):
    optimizer.zero_grad()
    output = model(data)
    loss = loss_func(output, target)
    loss.backward()
    optimizer.step()
    if local_rank == 0 and epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss {loss.item()}")

# Cleanup
dist.destroy_process_group()
