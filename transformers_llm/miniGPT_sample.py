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
