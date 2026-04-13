##Q3. Scaled Dot-Product Attention
import torch
import torch.nn.functional as F
import math

def attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    print("Before scaling:\n", scores)

    scaled_scores = scores / math.sqrt(d_k)
    print("\nAfter scaling:\n", scaled_scores)

    weights = F.softmax(scaled_scores, dim=-1)
    print("\nAttention Weights:\n", weights)

    output = torch.matmul(weights, V)
    return output

# Test with random tensors
Q = torch.rand(2, 4, 8)
K = torch.rand(2, 4, 8)
V = torch.rand(2, 4, 8)

output = attention(Q, K, V)

print("\nOutput:\n", output)
