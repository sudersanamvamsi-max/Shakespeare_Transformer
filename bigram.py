# IMPORT LIBRARIES
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
torch.manual_seed(1337)

# ---------- Hyperparameters ---------- #

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# --------------------------------------- #

# Inspect the dataset
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()
print(len(text))

#Get all the unique characters in the text /
# List of characters that the model will learn to predict (vocabulary)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Encode the text into integers
string_to_int = {ch:i for i, ch in enumerate(chars)} # Create a dictionary to map characters to integers
encode = lambda s: [string_to_int[c] for c in s] # Create a function to encode a string into a list of integers

# Decode the integers back into text
int_to_string = {i:ch for i, ch in enumerate(chars)} # Create a dictionary to map integers back to characters
decode = lambda l: ''.join([int_to_string[i] for i in l]) # Create a function to decode a list of integers into a string

print(encode("hello my name is vamsi"))
print(decode(encode("hello my name is vamsi")))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100]) # This is how the Transformer will see the data

n = int(0.9 * len(data)) #90% of the data for training, 10% for validation

train_data = data[:n]
val_data = data[n:]

# Block size is the number of characters that the model will see at a time
train_data[:block_size+1]

x = train_data[:block_size] # Input is the current character
y = train_data[1:block_size+1] # Target is the next character
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target is {target}")
 

def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)
print('--------------------------------')

# Now, lets start feeding this batch of input to feed to the transformer model
# We will start with the simplest possible neural network which is the bigram language model

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) where B is the batch size and T is the context length
        # C is the number of channels which i.e. vocabulary size
        logits = self.token_embedding_table(idx) # (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            # Apply Softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
logits, loss = model(xb, yb)
print(logits.shape)
print(loss)

print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# Lets start training the model
# Create a pytorch optimizer

# Smaller model means we can get away with a larger learning rate
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

for iter in range(max_iters):

    # Every once in a ehile evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")



    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(loss.item())

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))






