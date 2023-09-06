import torch
import torch.nn as nn
import torch.optim as optim

from time import time

start_time = time()

from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x = torch.rand((45,2)).to(device)
y = lambda x: (x[:, 0]**2)*3 + 4*x[:, 1] + 4
# y = torch.rand(45)
y = y(x).to(device)

class myModel(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        
        self.l1 = nn.Linear(input_dims, 1)
        
    def forward(self, x):
    
        x = self.l1(x)
        
        return x
    
model = myModel(x.shape[1]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

batch_size = 5

dataset = TensorDataset(x, y)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

epochs = 1_000

for epoch in range(epochs):
    
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()  # Zero the gradients

        y_pred = model(x_batch)      # Get predictions from the model

        loss = loss_fn(y_pred, y_batch.view(batch_size, -1))  # Compute the loss

        loss.backward()       # Backpropagate the loss

        optimizer.step()      # Update the model parameters

    if epoch % (epochs / 10) == 0:
        print(loss.item())
        
end_time = time()

full_time = end_time - start_time

print(f'time = {full_time}')
