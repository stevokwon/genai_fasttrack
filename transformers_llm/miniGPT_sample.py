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

# Making a mini Transformer
class MiniGPT(nn.module):
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, index):
        # index -> (Batch, Time)
        embed = self.embedding_table(index) # (Batch, Time, n_embed)
        logits = self.lm_head(embed) # (Batch, Time, vocab_size)
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
    x = torch.stack([data[i:i+block_size]] for i in ix)
    y = torch.stack([data[i+1:i+block_size]] for i in ix)

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
