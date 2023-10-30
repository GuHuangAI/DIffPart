import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttLayer(nn.Module):
    def __init__(self, dim, heads=4, ffn_dim_mul=2, drop=0.):
        super().__init__()
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.heads = heads
        # hidden_dim = dim_head * heads
        ffn_dim = ffn_dim_mul * dim
        self.softmax = nn.Softmax(dim=-1)
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.concat_lin = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, dim)
        )
        self.dropout = nn.Dropout(drop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, part_fea):
        b, l, c = x.shape
        l2 = part_fea.shape[1]
        q = self.q_lin(x)
        k = self.k_lin(part_fea)
        v = self.v_lin(part_fea)
        q = q.view(b, l, self.heads, self.dim_head).transpose(1, 2) # b, head, l, dim_head
        k = k.view(b, l2, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, l2, self.heads, self.dim_head).transpose(1, 2)

        q = q * self.scale
        k_t = k.transpose(2, 3)  # transpose
        att = self.softmax(q @ k_t)
        v = att @ v  # b, head, l ,dim_head
        v = v.transpose(1, 2).contiguous().view(b, l, c)
        x = x + self.concat_lin(v)
        x = self.norm1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x
class IndexMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, part_dim=128, n_layers=4):
        super(IndexMLP, self).__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.in_layer_part = nn.Linear(part_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - 2):
            self.hidden_layers.append(CrossAttLayer(hidden_dim))
        self.act = nn.ReLU()
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, part_fea):
        x = self.in_layer(x)
        part_fea = self.in_layer_part(part_fea)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x, part_fea)
        x = self.out_layer(self.act(x))
        return x

if __name__ == '__main__':
    model = IndexMLP(30, 10)
    x = torch.rand(1000, 1, 30)
    part_fea = torch.rand(1000, 8, 128)
    y = model(x, part_fea)
    pass