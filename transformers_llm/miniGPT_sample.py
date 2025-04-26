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

