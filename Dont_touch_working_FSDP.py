import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.optim import OSS
import torch.distributed as dist

from tqdm import trange

import os

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

# Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc(x)

def main():

    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Instantiate the model and wrap with FSDP
    model = SimpleModel()
    model = FSDP(model).cuda()

    # Use the sharded optimizer (OSS)
    optimizer = OSS(model.parameters(), lr=0.01, momentum=0.9)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in trange(50):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.cuda(), labels.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if local_rank == 0:
            print(f"Epoch {epoch+1}, Loss: {running_loss/i}")

    print("Finished Training!")

if __name__ == '__main__':
    main()
