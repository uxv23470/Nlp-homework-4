##Mini Transformer Encoder
import torch
import torch.nn as nn
import math

sentences = ["i love nlp", "transformers are powerful", "attention is key"]

# Tokenization
words = list(set(" ".join(sentences).split()))
word2idx = {w:i for i,w in enumerate(words)}

def encode(sentence):
    return [word2idx[w] for w in sentence.split()]

encoded = [encode(s) for s in sentences]
max_len = max(len(x) for x in encoded)

# Padding
for i in range(len(encoded)):
    encoded[i] += [0]*(max_len - len(encoded[i]))

X = torch.tensor(encoded)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        pe = torch.zeros(100, d_model)
        for pos in range(100):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**(i/d_model)))
                pe[pos, i+1] = math.cos(pos/(10000**(i/d_model)))
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer Encoder
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos(x)

        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x, attn_weights

model = MiniTransformer(len(words))
output, attn = model(X)

print("Input Tokens:\n", X)
print("\nFinal Embeddings:\n", output)
print("\nAttention Weights:\n", attn)
