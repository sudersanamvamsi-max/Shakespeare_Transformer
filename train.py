#Today we will learn how to code and train a transformer on .............. Shakespear!

# Open Dataset to train the model
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f'Length of dataset in characters: {len(text)}') 

# Taking a look at thr first 1500 characters
print(text[:1500])


# List of all unique characters that appear in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(f'Vocabulary size: {vocab_size}')

# Creating a mapping from characters to integers
# Encoding and Decoding functions

stoi = { ch:i for i,ch in enumerate(chars) } # stoi stands for string to integer, it is a dictionary that maps each character to a unique integer index
itos = { i:ch for i,ch in enumerate(chars) } # itos stands for integer to string, it is a dictionary that maps each integer index back to the corresponding character

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print (encode("hii there")) # example of encoding
print (decode(encode("hii there"))) # example of decoding

# Now we encode the entire dataset and store it in a torch tensor

import torch
data = torch.tensor(encode(text), dtype=torch.long) # we encode the entire text and convert it to a torch tensor
print(data.shape, data.dtype)
print(data[:1000])

# Splitting the data into train and test sets
n = int(0.85*len(data)) 
train_data = data[:n]
test_data = data[n:]

# Defining some hyperparameters for the model
block_size = 8
train_data[:block_size+1] 

x = train_data[:block_size] 
y = train_data[1:block_size+1] # off by 1 as we want to predict the next character
for t in range(block_size):
    context = x[:t+1] # the context grows as t increases
    target = y[t] # the target is the next character we want to predict
    print(f'When input is {context} the target: {target}')

# More Hyperparameters
torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data; inputs and targets for the model
    split_data = train_data if split == 'train' else test_data
    indices = torch.randint(len(split_data) - block_size, (batch_size,))
    inputs = torch.stack([split_data[i:i+block_size] for i in indices])
    targets = torch.stack([split_data[i+1:i+block_size+1] for i in indices])
    return inputs, targets

xb, yb = get_batch('train')

print('inputs:')
print(xb.shape)
print(xb)

print('targets:')
print(yb.shape)
print(yb)

print('@'*10)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f'When input is {context.tolist()} the target: {target}')

print(xb) # Input to the transformer

import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensors of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # (B*T, C)
            targets = targets.view(B*T) # (B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx, targets=None) # get the predictions
            logits = logits[:, -1, :] # focus only on the last time step
            probs = F.softmax(logits, dim=-1) # get probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample from the distribution
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence
        return idx
    
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape) # (Batch, Time, Channel) = (4, 8, 65)
print(loss)

print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist())) # generate from the model