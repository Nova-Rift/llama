import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

from accelerate import Accelerator
accelerator = Accelerator()
local_rank = accelerator.local_process_index

# hyperparaemters
n_embd = 64
n_layer = 4
n_head = 4
block_size = 64
batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
learning_rate = 1e-3

torch.manual_seed(1)

text = open('../data/Stories/full.txt', encoding='utf-8').read()
# text = open('../data/shakespeare.txt', encoding='utf-8').read()

chars = sorted(list(set(text)))
vocab_size = len(chars)


# create mapping functions
itos = { i: c for i, c in enumerate(chars) }
stoi = { c: i for i, c in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text))
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]



def get_batch(split):
    
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    X = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    
    X, y = X.to(device), y.to(device)
    return X, y

def get_all_data(split, small=None):
    data = train_data if split == 'train' else val_data
    if small == True:
        X = torch.stack([data[i*block_size:(i*block_size)+block_size] for i in range(len(data)//block_size)])
        y = torch.stack([data[(i*block_size)+1:(i*block_size)+block_size+1] for i in range(len(data)//block_size)])
    else:
        X = torch.stack([data[i:i+block_size] for i in range(len(data)-block_size)])
        y = torch.stack([data[i+1:i+1+block_size] for i in range(len(data)-block_size)])
    X, y = X.to(device), y.to(device)
    return X, y

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        
        B,T,C = x.shape
        
        k = self.key(x)
        q = self.query(x)
        # compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        # value
        v = self.value(x)
        out = wei @ v
        return out
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        return x
    
class FeedForward(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x
    
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        
        head_size = n_embd // n_head
        
        self.sa = MultiHeadAttention(head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.ln1(self.sa(x))
        x = x + self.ln2(self.ffwd(x))
        return x
    
class Transformer(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T).to(device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets == None:
            loss = None
            
        else:
            
            B,T,C = logits.shape
            
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
        
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx        
        
        
model = Transformer().to(device)
print(sum(p.numel() for p in model.parameters(), 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

X, y = get_all_data('train', True)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

epochs = 1
for epoch in range(epochs):
    
    for i, (xb, yb) in enumerate(dataloader):

        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        accelerator.backward(loss)
#         loss.backward()
        optimizer.step()
        
        if (i % (len(dataloader) // 10) == 0 or i == 0) and local_rank ==0:
            data = estimate_loss()
            print(f"epoch {epoch} : train loss {data['train']} : val loss {data['val']}")        
    
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
if local_rank == 0:
    print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))