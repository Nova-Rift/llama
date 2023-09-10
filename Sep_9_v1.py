import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1)

# hyperparameters
batch_size = 16

X = torch.rand(4096, 3).to(device)
# y_transform = lambda x: (x[:, 0]**3)*3 + (x[:, 1]**2)*7 + x[:, 2]
y_transform = lambda x: (
    (3 * x[:, 0]**4 * torch.sin(x[:, 0]**2 / (x[:, 1] + 1)) +
     (7 * x[:, 1]**3 / (1 + torch.exp(-x[:, 1] + x[:, 0]))) * torch.log(x[:, 1]**2 + x[:, 0]) * torch.tan(x[:, 2]**2 / (x[:, 0] + 1)) -
     x[:, 2]**3 * torch.cos(x[:, 2]**2) * torch.exp(x[:, 0] - x[:, 1]))
    *
    (2 + torch.cos(x[:, 0] + x[:, 1]**2 - x[:, 2]) + torch.log(1 + x[:, 2]))
)
y = y_transform(X).view(X.shape[0], 1)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class myModel(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        
        dims = 1024
        
        self.l1 = nn.Linear(input_dims, dims)
        self.lr1 = nn.ReLU()
        self.l2 = nn.Linear(dims, dims)
        self.lr2 = nn.ReLU()
        self.l3 = nn.Linear(dims, 1)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.lr1(x)
        x = self.l2(x)
        x = self.lr2(x)
        x = self.l3(x)
        
        return x
    
    
model = myModel(X.shape[1]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

epochs = 20

for epoch in range(epochs):
    
    for X_batch, y_batch in dataloader:
    
        y_pred = model(X_batch)

        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % (epochs / 10) == 0:
        print(loss.item())
        

test_tensor = torch.tensor([0.3, 0.1, 0.9]).view(1,3).to(device)
        
print(f'test = {y_transform(test_tensor).item()}')
print(f'true = {model(test_tensor).item()}')