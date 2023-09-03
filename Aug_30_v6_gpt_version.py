import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, TensorDataset

# Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

def main():
    # Initialize distributed environment
    dist.init_process_group(backend="nccl")
    
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

    # Training loop
    for epoch in range(50):
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

if __name__ == "__main__":
    main()
