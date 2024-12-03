import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_every = 100
eval_iters = 200
learning_rate = 1e-3
n_embed = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data: wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Length of dataset in chars: ", len(text))

# build the vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}
# encoder: take a string and output a list of integers
encode = lambda s: [stoi[c] for c in s]
# decoder: take a list of integers and output a string
decode = lambda l: ''.join([itos[e] for e in l])
assert decode(encode('hello')) == 'hello'

# split the train and val data
data = torch.tensor(encode(text))
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
print("Train and val data shapes: ", train_data.shape, val_data.shape)

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    # estimate the loss on the validation set
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = m(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class Head(nn.Module):
    """ one head of self attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x) # (B, T, head_size)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, T)
        # zero out the upper triangular part of the matrix
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # apply the attention to the values
        out = wei @ v # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple parallel heads of self attention """

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ a transformer block of communication and computation interspersed """

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = self.sa(self.ln1(x)) + x
        x = self.ffwd(self.ln2(x)) + x
        return x

# build the model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            nn.LayerNorm(n_embed),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are of shape (B, T)
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C) using broadcasting for addition
        x = self.blocks(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # reshaping to (B*T, C) as torch expects C to be next to mini-batch
            targets = targets.view(B*T) # reshaping to (B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # generate(...) should take (B, T) and generate (B, T+1),
        # take that and generate (B, T+2) until max_new_tokens
        for _ in range(max_new_tokens):
            # crop the context to the last block_size tokens
            idx = idx[:, -block_size:]
            # get the preds
            logits, _ = self(idx)
            # focus only on the last time step because it is bigram
            logits = logits[:, -1, :] # (B, C)
            # apply softmax
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel()

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):

    # eval on the validation set every eval_every iterations
    if iter % eval_every == 0:
        losses = estimate_loss()
        print(f"iter {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    
    # get batch
    xb, yb = get_batch('train')
    # forward pass
    logits, loss = m(xb, yb)
    # zero grad
    optimizer.zero_grad(set_to_none=True)
    # backward pass
    loss.backward()
    # parameter update
    optimizer.step()

# generate some text
context = torch.zeros((1, 1), dtype=torch.long)
print("Generating text: ")
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))