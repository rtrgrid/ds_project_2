import torch
import torch.nn as nn
import math


# ----------------------------
# Positional Encoding (Sinusoidal)
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ----------------------------
# ALiBi Bias
# ----------------------------
class ALiBiBias(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.slopes = torch.arange(1, n_heads + 1).float()

    def forward(self, seq_len):
        bias = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        bias = bias.unsqueeze(0) * self.slopes.view(-1, 1, 1)
        return bias


# ----------------------------
# Custom Encoder Layer
# ----------------------------
class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, use_alibi=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_alibi = use_alibi

        # projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        # feedforward
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(0.1)

        # ALiBi
        if self.use_alibi:
            self.alibi = ALiBiBias(nhead)

    def forward(self, x):
        B, T, D = x.shape

        # projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 🔥 ALiBi
        if self.use_alibi:
            bias = self.alibi(T).to(x.device)   # [heads, T, T]
            scores = scores + bias.unsqueeze(0)

        weights = torch.softmax(scores, dim=-1)
        self.attn_weights = weights  # 🔥 store attention
        attn_output = torch.matmul(weights, v)

        # merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        attn_output = self.out_proj(attn_output)

        # residual + norm
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # feedforward
        ff = self.linear2(torch.relu(self.linear1(x)))

        x = x + self.dropout(ff)
        x = self.norm2(x)

        return x


# ----------------------------
# Transformer Model
# ----------------------------
class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        horizon=12,
        positional_type="sinusoidal"
    ):
        super().__init__()
        print("🔥 Using NEW Custom Transformer Model")

        self.positional_type = positional_type

        # input
        self.input_projection = nn.Linear(input_size, d_model)

        # positional encoding
        if positional_type == "sinusoidal":
            self.positional_encoding = PositionalEncoding(d_model)
            use_alibi = False

        elif positional_type == "learnable":
            self.pos_embedding = nn.Parameter(torch.randn(1, 500, d_model))
            use_alibi = False

        elif positional_type == "none":
            self.positional_encoding = None
            use_alibi = False

        elif positional_type == "alibi":
            self.positional_encoding = None
            use_alibi = True

        else:
            raise ValueError("Invalid positional_type")

        # transformer layers
        self.transformer = nn.ModuleList([
            CustomEncoderLayer(d_model, nhead, use_alibi=use_alibi)
            for _ in range(num_layers)
        ])

        # output
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.input_projection(x)

        # positional
        if self.positional_type == "sinusoidal":
            x = self.positional_encoding(x)

        elif self.positional_type == "learnable":
            x = x + self.pos_embedding[:, :x.size(1), :]

        # ALiBi / none → no addition

        # transformer
        for layer in self.transformer:
            x = layer(x)

        x = x[:, -1, :]
        return self.fc(x)