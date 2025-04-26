import torch
import torch.nn as nn
import torch.nn.functional as f

# Tiny dataset
text = "Hello World"

# Create character-level vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Character to Index Mapping
stoi = {ch: i for i, ch in enumerate(chars)} # String to Integer
itos = {i: ch for ch, i in stoi.items()} # Integer to String

# Encoder and Decoder functions
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join ([itos[i] for i in l])

# Training Data
data = torch.tensor(encode(text), dtype = torch.long)

# Create a Tiny Self-Attention Head
class SelfAttention(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.key = nn.Linear(n_embed, n_embed, bias = False)
        self.query = nn.Linear(n_embed, n_embed, bias = False)
        self.value = nn.Linear(n_embed, n_embed, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(1024, 1024)))
    
    def forward(self, x):
        B, T, C = x.shape # Batch, Time, Channels

        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # Compute attention scores -> 'affinities'
        wei = q @ k.transpose(-2, -1) * C**0.5 # (B, T, T)

        # Mask out future tokens (causal attention)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Softmax to get attention weight
        wei = f.softmax(wei, dim = -1) # (B, T, T)

        # Weighted aggregation
        v = self.value(x)
        out = wei @ v

        return out

# Mini Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.attn = SelfAttention(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed)
        )
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x += self.attn(self.ln1(x)) # Residual connection after attention
        x += self.mlp(self.ln2(x)) # Residual connection after MLP
        return x

# Making a mini Transformer
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embed)
        self.transformer = TransformerBlock(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, index):
        embed = self.embedding_table(index) # (Batch, Time, n_embed)
        out = self.transformer(embed) # (Batch, Time, n_embed)
        logits = self.lm_head(out) # (Batch, Time, vocab_size)
        return logits

# Model Hyperparameter
n_embed = 32 # Embedding Size

model = MiniGPT(vocab_size, n_embed)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-2)

# Training Loop
batch_size = 4
block_size = 4 # Context Length

for step in range(300):
    # get random batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # Forward
    logits = model(x)
    logits = logits.view(-1, vocab_size)
    y = y.view(-1)
    loss = f.cross_entropy(logits, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Loss : {loss.item():.4f}")

# Start with context of 1 character
context = torch.zeros((1, 1), dtype = torch.long)

# Generate 100 characters
generated = []
for _ in range(100):
    logits = model(context)
    logits = logits[:, -1, :] # Take the last time step
    probs = f.softmax(logits, dim = -1)
    idx_next = torch.multinomial(probs, num_samples = 1)
    generated.append(idx_next.item())
    context = torch.cat((context, idx_next), dim = 1)

print(decode(generated))
