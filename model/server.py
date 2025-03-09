import torch
import torch.nn as nn
from flask import Flask, request, jsonify

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        forward_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(forward_output))
        return x

embed_size = 128
heads = 8
dropout = 0.1
forward_expansion = 4
model = TransformerBlock(embed_size, heads, dropout, forward_expansion)
model.load_state_dict(torch.load("transformer.pth"))

model.eval()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    with torch.no_grad():
        output = model(torch.tensor(data["input"]).float().unsqueeze(1))
    return jsonify({"output": output.squeeze(1).tolist()})

if __name__ == "__main__":
    app.run(debug=True)

