import torch

torch.manual_seed(1337)
B, T, C = 4,8,2 # Batch, Time, Channel 
x = torch.randn(B, T, C)
x.shape

# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B,T,C)) # bow = bag of words
for b in range(B):
    for t in range (T):
        xprev = x[b,:t+1] #(t,C)
        xbow[b,t] = torch.mean(xprev, 0)
