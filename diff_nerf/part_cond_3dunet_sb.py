import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce
from functools import partial
import numpy as np

### uncond 3d unet for single branch ###
### use part feature as condition

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

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

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv3d(dim, default(dim_out, dim), 3, 2, 1)

class WeightStandardizedConv3d(nn.Conv3d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv3d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv3d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)
        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h, scale_shift=scale_shift)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

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
                 window_size_k=[[4, 4, 4], [2, 2, 2], [1, 1, 1]], drop=0.1):
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
        self.out_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim)
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

class CondAttention(nn.Module): # Multi-Scale Window-Attention
    def __init__(self, dim, heads=4, window_size_q=[4, 4, 4],
                 window_size_k=[[4, 4, 4], [2, 2, 2], [1, 1, 1]], drop=0.1):
        super(CondAttention, self).__init__()
        # assert  dim == heads * dim_head
        dim_head = dim // heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        # self.qkv = nn.Conv3d(dim, hidden_dim*3, 1)
        self.q_lin = nn.Linear(dim, hidden_dim)
        self.k_lin = nn.Linear(dim, hidden_dim)
        self.v_lin = nn.Linear(dim, hidden_dim)
        self.pos_enc = PositionEmbeddingSine3D(hidden_dim)
        self.window_size_q = window_size_q
        self.avgpool_q = nn.AdaptiveAvgPool3d(output_size=window_size_q)
        # self.avgpool_ks = nn.ModuleList()
        # for i in range(len(window_size_k)):
        #     self.avgpool_ks.append(nn.AdaptiveAvgPool3d(output_size=window_size_k[i]))
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=hidden_dim*2, drop=drop)
        self.out_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim)
        )

    def forward(self, x, cond):
        # x: B, C, X, Y, Z
        # cond: num_parts, C
        B, C, X, Y, Z = x.shape
        shortcut = x
        q_s = self.avgpool_q(x)
        length = q_s.shape[-3] * q_s.shape[-2] * q_s.shape[-1]
        kg = cond.unsqueeze(0).expand(B, -1, -1)
        qg = self.avgpool_q(x).permute(0, 2, 3, 4, 1).contiguous()
        qg = qg + self.pos_enc(qg)
        qg = qg.view(B, -1, C)
        # kgs = []
        # for avgpool in self.avgpool_ks:
        #     kg_tmp = avgpool(x).permute(0, 2, 3, 4, 1).contiguous()
        #     kg_tmp = kg_tmp + self.pos_enc(kg_tmp)
        #     kg_tmp = kg_tmp.view(B, -1, C)
        #     kgs.append(kg_tmp)
        # kg = torch.cat(kgs, dim=1)

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

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class SpatialAtt(nn.Module):
    def __init__(self, in_dim):
        super(SpatialAtt, self).__init__()
        self.map = nn.Conv3d(in_dim, 1, 1)
        self.q_conv = nn.Conv3d(1, 1, 1)
        self.k_conv = nn.Conv3d(1, 1, 1)
        self.activation = nn.Softsign()

    def forward(self, x):
        b, _, h, w, z = x.shape
        att = self.map(x) # b, 1, h, w, z
        q = self.q_conv(att) # b, 1, h, w, z
        q = rearrange(q, 'b c h w z -> b (h w z) c')
        k = self.k_conv(att)
        k = rearrange(k, 'b c h w z -> b c (h w z)')
        att = rearrange(att, 'b c h w z -> b (h w z) c')
        att = F.softmax(q @ k, dim=-1) @ att # b, hw, 1
        att = att.reshape(b, 1, h, w, z)
        return self.activation(att) * x

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# model

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        text_dim= 128,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        heads=8,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        window_size_q=[8, 8, 8],
        window_size_k=[[8, 8, 8], [4, 4, 4], [2, 2, 2]],
        out_mul=1,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(input_channels, init_dim, 3, padding = 1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        dims_rev = dims[::-1]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            # sinu_pos_emb = SinusoidalPosEmb(dim)
            sinu_pos_emb = GaussianFourierProjection(dim//2)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # cond embedding
        self.cond_mlps = nn.ModuleList([])
        self.cond_mlps.append(nn.Linear(text_dim, dims[0]))
        self.cond_mlps.append(nn.Linear(text_dim, dims[1]))
        self.cond_mlps.append(nn.Linear(text_dim, dims[2]))
        self.cond_mlps.append(nn.Linear(text_dim, dims[3]))

        # layers

        self.downs = nn.ModuleList([])
        self.relations_down = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.relations_up = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                # Residual(PreNorm(dim_in, LinearAttention(dim_in, heads=heads))),
                MSWA_3D(dim=dim_in, heads=heads, window_size_q=window_size_q[ind], window_size_k=window_size_k[ind]),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding = 1)
            ]))
            self.relations_down.append(
                CondAttention(dim=dims[ind], heads=heads, window_size_q=window_size_q[len(in_out) - 1 - ind],
                        window_size_k=window_size_k[len(in_out) - 1 - ind]),
            )


        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        # self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, heads=heads)))
        self.mid_attn = MSWA_3D(dim=mid_dim, heads=heads, window_size_q=window_size_q[-1], window_size_k=window_size_k[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.decouple1 = nn.Sequential(
            nn.GroupNorm(num_groups=min(mid_dim // 4, 8), num_channels=mid_dim),
            nn.Conv3d(mid_dim, mid_dim, 1, padding=0),
            SpatialAtt(mid_dim))
        # self.decouple2 = nn.Sequential(
        #     nn.GroupNorm(num_groups=min(mid_dim // 4, 8), num_channels=mid_dim),
        #     nn.Conv3d(mid_dim, mid_dim, 1, padding=0),
        #     SpatialAtt(mid_dim))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                # Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                MSWA_3D(dim=dim_out, heads=heads, window_size_q=window_size_q[len(in_out) - 1 - ind],
                                                window_size_k=window_size_k[len(in_out) - 1 - ind]),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv3d(dim_out, dim_in, 3, padding = 1)
            ]))
            self.relations_up.append(
                CondAttention(dim=dims_rev[ind], heads=heads, window_size_q=window_size_q[len(in_out) - 1 - ind],
                              window_size_k=window_size_k[len(in_out) - 1 - ind]),
            )
            # self.ups2.append(nn.ModuleList([
            #     block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
            #     block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
            #     # Residual(PreNorm(dim_out, LinearAttention(dim_out))),
            #     MSWA_3D(dim=dim_out, heads=heads, window_size_q=window_size_q[len(in_out) - 1 - ind],
            #                                     window_size_k=window_size_k[len(in_out) - 1 - ind]),
            #     Upsample(dim_out, dim_in) if not is_last else nn.Conv3d(dim_out, dim_in, 3, padding=1)
            # ]))

        default_out_dim = channels * out_mul
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1)

        # self.final_res_block2 = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        # self.final_conv2 = nn.Conv3d(dim, self.out_dim, 1)

    def forward(self, x, time, cond=None, x_self_cond=None, *args, **kwargs): ## cond is always None for unconditional model
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        sigma = time.reshape(-1, 1, 1, 1, 1)
        # c_skip1 = 1 - sigma
        # c_skip2 = torch.sqrt(sigma)
        # c_out1 = sigma / torch.sqrt(sigma ** 2 + 1)
        # c_out2 = torch.sqrt(1 - sigma) / torch.sqrt(sigma ** 2 + 1)
        c_in = 1 / (1 + sigma).sqrt()
        c_noise = time.log()

        x_clone = x.clone()
        x = c_in * x
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(c_noise)

        part_embs = []
        for i, layer in enumerate(self.cond_mlps):
            part_embs.append(layer(cond))

        h = []
        # h2 = []

        for i, ((block1, block2, attn, downsample), relation_layer) in \
                enumerate(zip(self.downs, self.relations_down)):
            x = block1(x, t)
            h.append(x)
            # h2.append(x.clone())
            x = relation_layer(x, part_embs[i])
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            # h2.append(x.clone())

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x1 = x + self.decouple1(x)
        # x2 = x + self.decouple2(x)

        x = x1
        for (block1, block2, attn, upsample), relation_layer in zip(self.ups, self.relations_up):
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = relation_layer(x, part_embs.pop())
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)
        x1 = torch.cat((x, r), dim = 1)
        x1 = self.final_res_block(x1, t)
        x1 = self.final_conv(x1)

        # x = x2
        # for block1, block2, attn, upsample in self.ups2:
        #     x = torch.cat((x, h2.pop()), dim=1)
        #     x = block1(x, t)
        #
        #     x = torch.cat((x, h2.pop()), dim=1)
        #     x = block2(x, t)
        #     x = attn(x)
        #
        #     x = upsample(x)
        # x2 = torch.cat((x, r), dim=1)
        # x2 = self.final_res_block2(x2, t)
        # x2 = self.final_conv2(x2)

        # x1 = c_skip1 * x_clone + c_out1 * x1
        # x2 = c_skip2 * x_clone + c_out2 * x2
        x2 = (x_clone - (sigma - 1) * x1) / sigma.sqrt()
        return x1, x2

if __name__ == '__main__':
    # x = torch.rand(1, 16, 16, 16, 128)
    # pe = PositionEmbeddingSine3D(128, normalize=True)
    # pe_x = pe(x)
    # mswa = MSWA_3D(dim=128, heads=8, dim_head=16)
    # z = mswa(x.permute(0, 4, 1, 2, 3))

    model = Unet3D(64, channels=3, out_mul=1, dim_mults=[1,2,4,8], heads=8,
                   window_size_q=[[8, 8, 8], [4, 4, 4], [2, 2, 2], [1, 1, 1]],
                   window_size_k=[[[8, 8, 8], [4, 4, 4], [2, 2, 2], [1, 1, 1]],
                                  [[4, 4, 4], [2, 2, 2], [1, 1, 1]],
                                  [[2, 2, 2], [1, 1, 1]],
                                  [[1, 1, 1]],])
    x = torch.rand(2, 3, 16, 16, 16)
    time = torch.tensor([0.2567, 0.5284])
    cond = torch.rand(4, 128)
    with torch.no_grad():
        y = model(x, time, cond)
        pass