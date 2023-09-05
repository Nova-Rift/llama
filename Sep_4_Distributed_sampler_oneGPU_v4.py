import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Generate structured data
input_size = 10
num_samples = 100
X = torch.rand(num_samples, input_size)
y = X.sum(dim=1, keepdim=True)

# Create DataLoader
batch_size = 4
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size)


# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        dims = 30720 #28672 works
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, dims)
        self.fc2 = nn.Linear(dims, dims)
        self.fc3 = nn.Linear(dims, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = SimpleNN(input_size).cuda()

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00005)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):

    for i, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(inputs.cuda())

        loss = criterion(outputs, targets.cuda())

        loss.backward()

        optimizer.step()

    if (epoch+1) % 1 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

print('Training complete')
