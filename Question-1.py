##Q1. Character-Level RNN Language Model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

text = "hello help hello help hello hero hello hill"
chars = sorted(list(set(text)))
char2idx = {ch:i for i,ch in enumerate(chars)}
idx2char = {i:ch for ch,i in char2idx.items()}

data = [char2idx[ch] for ch in text]

seq_length = 10

def get_batch(data, seq_length):
    inputs = []
    targets = []
    for i in range(len(data) - seq_length):
        inputs.append(data[i:i+seq_length])
        targets.append(data[i+1:i+seq_length+1])
    return torch.tensor(inputs), torch.tensor(targets)

X, Y = get_batch(data, seq_length)

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

model = CharRNN(len(chars))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.view(-1, len(chars)), Y.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

def sample(model, start="h", length=200, temp=1.0):
    model.eval()
    input_char = torch.tensor([[char2idx[start]]])
    hidden = None
    result = start

    for _ in range(length):
        output = model(input_char)
        logits = output[0, -1] / temp
        probs = torch.softmax(logits, dim=0).detach().numpy()
        idx = np.random.choice(len(chars), p=probs)
        result += idx2char[idx]
        input_char = torch.tensor([[idx]])

    return result

print("\nTemp 0.7:\n", sample(model, temp=0.7))
print("\nTemp 1.0:\n", sample(model, temp=1.0))
print("\nTemp 1.2:\n", sample(model, temp=1.2))
