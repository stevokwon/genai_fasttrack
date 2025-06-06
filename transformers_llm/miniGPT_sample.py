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
    def __init__(self, n_embed, n_heads = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim = n_embed, num_heads = n_heads, batch_first = True)
    
    def forward(self, x):
        # Causal mask: prevent attending to future tokens
        T = x.size(1)
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)

        out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        return out

# Mini Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, n_embed, dropout = 0.1):
        super().__init__()
        self.attn = SelfAttention(n_embed, n_heads = 4)
        self.ln1 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        attn_out = self.attn(self.ln1(x)) # Self-attention output
        x = x + attn_out # Out-of-place residual connection

        mlp_out = self.mlp(self.ln2(x)) # MLP output
        x = x + mlp_out # Out-of-place residual connection
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
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)

# Training Loop
batch_size = 4
block_size = 4 # Context Length

for step in range(500):
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
