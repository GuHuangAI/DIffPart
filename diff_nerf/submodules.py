import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        # self.class_token_pos = nn.Parameter(torch.zeros(1, 1, num_pos_feats * 2))
        # self.class_token_pos

    def forward(self, x):
        # x: b, x, y, z, d
        num_feats = x.shape[-1]
        num_pos_feat = num_feats // 3
        num_pos_feats = [num_pos_feat, num_pos_feat, num_feats - 2 * num_pos_feat]
        # mask = tensor_list.mask
        mask = torch.zeros(*(x.shape[:4]), device=x.device).to(torch.bool)
        batch = mask.shape[0]
        assert mask is not None
        not_mask = ~mask
        z_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-5
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(num_pos_feats[0], dtype=torch.float32, device=x.device)
        dim_tx = self.temperature ** (2 * (dim_tx // 2) / num_pos_feats[0])
        dim_ty = torch.arange(num_pos_feats[1], dtype=torch.float32, device=x.device)
        dim_ty = self.temperature ** (2 * (dim_ty // 2) / num_pos_feats[1])
        dim_tz = torch.arange(num_pos_feats[2], dtype=torch.float32, device=x.device)
        dim_tz = self.temperature ** (2 * (dim_tz // 2) / num_pos_feats[2])


        pos_x = x_embed[:, :, :, :, None] / dim_tx
        pos_y = y_embed[:, :, :, :, None] / dim_ty
        pos_z = z_embed[:, :, :, :, None] / dim_tz
        pos_x = torch.cat((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=-1).flatten(4)
        pos_y = torch.cat((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=-1).flatten(4)
        pos_z = torch.cat((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=-1).flatten(4)
        # pos = torch.cat((pos_y, pos_x), dim=3).flatten(1, 2)
        pos = torch.cat((pos_z, pos_y, pos_x), dim=4).contiguous()
        '''
        pos_x: b, x, y, z, d//3
        pos_y: b, x, y, z, d//3
        pos_z: b, x, y, z, d//3
        pos: b, x, y, z, d
        '''
        return pos

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Conv3d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, kernel_size=1)
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MSWA_3D(nn.Module): # Multi-Scale Window-Attention
    def __init__(self, dim, heads=4, window_size_q=[4, 4, 4],
                 window_size_k=[[4, 4, 4], [2, 2, 2], [1, 1, 1]], drop=0.1, groups=1):
        super(MSWA_3D, self).__init__()
        # assert  dim == heads * dim_head
        dim_head = dim // heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        # self.qkv = nn.Conv3d(dim, hidden_dim*3, 1)
        self.q_lin = nn.Linear(dim, hidden_dim, 1)
        self.k_lin = nn.Linear(dim, hidden_dim, 1)
        self.v_lin = nn.Linear(dim, hidden_dim, 1)
        self.pos_enc = PositionEmbeddingSine3D(hidden_dim)
        self.window_size_q = window_size_q
        self.avgpool_q = nn.AdaptiveAvgPool3d(output_size=window_size_q)
        self.avgpool_ks = nn.ModuleList()
        for i in range(len(window_size_k)):
            self.avgpool_ks.append(nn.AdaptiveAvgPool3d(output_size=window_size_k[i]))
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=hidden_dim*2, drop=drop)
        num_g = 8 if hidden_dim > 8 else 2
        self.out_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 1, groups=groups),
            nn.GroupNorm(num_g, hidden_dim)
        )

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        shortcut = x
        q_s = self.avgpool_q(x)
        qg = self.avgpool_q(x).permute(0, 2, 3, 4, 1).contiguous()
        qg = qg + self.pos_enc(qg)
        qg = qg.view(B, -1, C)
        kgs = []
        for avgpool in self.avgpool_ks:
            kg_tmp = avgpool(x).permute(0, 2, 3, 4, 1).contiguous()
            kg_tmp = kg_tmp + self.pos_enc(kg_tmp)
            kg_tmp = kg_tmp.view(B, -1, C)
            kgs.append(kg_tmp)
        kg = torch.cat(kgs, dim=1)

        num_window_q = qg.shape[1]
        num_window_k = kg.shape[1]
        qg = self.q_lin(qg).reshape(B, num_window_q, self.heads, C // self.heads).permute(0, 2, 1,
                                                                                           3).contiguous()
        kg2 = self.k_lin(kg).reshape(B, num_window_k, self.heads, C // self.heads).permute(0, 2, 1,
                                                                                            3).contiguous()
        vg = self.v_lin(kg).reshape(B, num_window_k, self.heads, C // self.heads).permute(0, 2, 1,
                                                                                           3).contiguous()
        kg = kg2
        attn = (qg @ kg.transpose(-2, -1))
        attn = self.softmax(attn)
        qg = (attn @ vg).transpose(1, 2).reshape(B, num_window_q, C)
        qg = qg.transpose(1, 2).reshape(B, C, self.window_size_q[0], self.window_size_q[1], self.window_size_q[2])
        # qg = F.interpolate(qg, size=(H1p, W1p), mode='bilinear', align_corners=False)
        q_s = q_s + qg
        q_s = q_s + self.mlp(q_s)
        q_s = F.interpolate(q_s, size=(X, Y, Z), mode='trilinear', align_corners=True)
        out = shortcut + self.out_conv(q_s)
        return out


if __name__ == '__main__':
    model = IndexMLP(30, 10)
    x = torch.rand(1000, 1, 30)
    part_fea = torch.rand(1000, 8, 128)
    y = model(x, part_fea)
    pass