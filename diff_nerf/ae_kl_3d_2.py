# pytorch_diffusion + one mlp + multi-scale attention feature grid  + optimizing together + independent loss

# one std_scale
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from diff_nerf.loss import LPIPSWithDiscriminator
import os
from diff_nerf.data import max_min_unnormalize
from tqdm.auto import tqdm
from train_cls_3d import generate_model
from torch_scatter import segment_coo
from diff_nerf import dvgo, grid
from scipy import ndimage
from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)
total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

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

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, groups=1):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=groups)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, groups=1):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0,
                                        groups=groups)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1, 0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, groups=1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     groups=groups)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     groups=groups)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     groups=groups)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    groups=groups)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1).contiguous()   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1).contiguous()   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True, use_linear_attn=False, attn_type="vanilla"):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t=None, context=None):
        #assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="mswa",
                 groups=1,**ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv3d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       groups=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         groups=groups))
                block_in = block_out
                if curr_res in attn_resolutions:
                    # attn.append(make_attn(block_in, attn_type=attn_type))
                    attn.append(MSWA_3D(dim=block_in, heads=4,
                                    window_size_q=[8, 8, 8],
                                    window_size_k=[[8, 8, 8], [4, 4, 4], [2, 2, 2]],
                                    groups=groups))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv, groups=groups)
                curr_res = (curr_res[0] // 2, curr_res[1] // 2)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       groups=groups)
        # self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.attn_1 = MSWA_3D(dim=block_in, heads=4,
                                window_size_q=[8, 8, 8],
                                window_size_k=[[8, 8, 8], [4, 4, 4], [2, 2, 2]],
                                groups=groups)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       groups=groups)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 groups=1, attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = (resolution[0] // 2**(self.num_resolutions-1),
                    resolution[1] // 2**(self.num_resolutions-1),
                    resolution[2] // 2**(self.num_resolutions-1))
        self.z_shape = (1,z_channels,curr_res[0],curr_res[1], curr_res[2])
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv3d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       groups=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       groups=groups)
        # self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.attn_1 = MSWA_3D(dim=block_in, heads=4,
                                  window_size_q=[8, 8, 8],
                                  window_size_k=[[8, 8, 8], [4, 4, 4], [2, 2, 2]],
                                  groups=groups)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       groups=groups)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         groups=groups))
                block_in = block_out
                if curr_res in attn_resolutions:
                    # attn.append(make_attn(block_in, attn_type=attn_type))
                    attn.append(MSWA_3D(dim=block_in, heads=4,
                                  window_size_q=[8, 8, 8],
                                  window_size_k=[[8, 8, 8], [4, 4, 4], [2, 2, 2]],
                                  groups=groups))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv, groups=groups)
                curr_res = (curr_res[0] * 2, curr_res[1] * 2)
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1),
                                     ResnetBlock(in_channels=in_channels,
                                                 out_channels=2 * in_channels,
                                                 temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=2 * in_channels,
                                                out_channels=4 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=4 * in_channels,
                                                out_channels=2 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     nn.Conv2d(2*in_channels, in_channels, 1),
                                     Upsample(in_channels, with_conv=True)])
        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1,2,3]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2,2), dropout=0.0):
        super().__init__()
        # upsampling
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # upsampling
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class LatentRescaler(nn.Module):
    def __init__(self, factor, in_channels, mid_channels, out_channels, depth=2):
        super().__init__()
        # residual block, interpolate, residual block
        self.factor = factor
        self.conv_in = nn.Conv2d(in_channels,
                                 mid_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.res_block1 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])
        self.attn = AttnBlock(mid_channels)
        self.res_block2 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])

        self.conv_out = nn.Conv2d(mid_channels,
                                  out_channels,
                                  kernel_size=1,
                                  )

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.res_block1:
            x = block(x, None)
        x = torch.nn.functional.interpolate(x, size=(int(round(x.shape[2]*self.factor)), int(round(x.shape[3]*self.factor))))
        x = self.attn(x)
        for block in self.res_block2:
            x = block(x, None)
        x = self.conv_out(x)
        return x


class MergedRescaleEncoder(nn.Module):
    def __init__(self, in_channels, ch, resolution, out_ch, num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 ch_mult=(1,2,4,8), rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        intermediate_chn = ch * ch_mult[-1]
        self.encoder = Encoder(in_channels=in_channels, num_res_blocks=num_res_blocks, ch=ch, ch_mult=ch_mult,
                               z_channels=intermediate_chn, double_z=False, resolution=resolution,
                               attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=resamp_with_conv,
                               out_ch=None)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=intermediate_chn,
                                       mid_channels=intermediate_chn, out_channels=out_ch, depth=rescale_module_depth)

    def forward(self, x):
        x = self.encoder(x)
        x = self.rescaler(x)
        return x


class MergedRescaleDecoder(nn.Module):
    def __init__(self, z_channels, out_ch, resolution, num_res_blocks, attn_resolutions, ch, ch_mult=(1,2,4,8),
                 dropout=0.0, resamp_with_conv=True, rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        tmp_chn = z_channels*ch_mult[-1]
        self.decoder = Decoder(out_ch=out_ch, z_channels=tmp_chn, attn_resolutions=attn_resolutions, dropout=dropout,
                               resamp_with_conv=resamp_with_conv, in_channels=None, num_res_blocks=num_res_blocks,
                               ch_mult=ch_mult, resolution=resolution, ch=ch)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=z_channels, mid_channels=tmp_chn,
                                       out_channels=tmp_chn, depth=rescale_module_depth)

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Upsampler(nn.Module):
    def __init__(self, in_size, out_size, in_channels, out_channels, ch_mult=2):
        super().__init__()
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size//in_size))+1
        factor_up = 1.+ (out_size % in_size)
        print(f"Building {self.__class__.__name__} with in_size: {in_size} --> out_size {out_size} and factor {factor_up}")
        self.rescaler = LatentRescaler(factor=factor_up, in_channels=in_channels, mid_channels=2*in_channels,
                                       out_channels=in_channels)
        self.decoder = Decoder(out_ch=out_channels, resolution=out_size, z_channels=in_channels, num_res_blocks=2,
                               attn_resolutions=[], in_channels=None, ch=in_channels,
                               ch_mult=[ch_mult for _ in range(num_blocks)])

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Resize(nn.Module):
    def __init__(self, in_channels=None, learned=False, mode="bilinear"):
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        if self.with_conv:
            print(f"Note: {self.__class__.__name} uses learned downsampling and will ignore the fixed {mode} mode")
            raise NotImplementedError()
            assert in_channels is not None
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)

    def forward(self, x, scale_factor=1.0):
        if scale_factor==1.0:
            return x
        else:
            x = torch.nn.functional.interpolate(x, mode=self.mode, align_corners=False, scale_factor=scale_factor)
        return x

class FirstStagePostProcessor(nn.Module):

    def __init__(self, ch_mult:list, in_channels,
                 pretrained_model:nn.Module=None,
                 reshape=False,
                 n_channels=None,
                 dropout=0.,
                 pretrained_config=None):
        super().__init__()
        if pretrained_config is None:
            assert pretrained_model is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.pretrained_model = pretrained_model
        else:
            assert pretrained_config is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.instantiate_pretrained(pretrained_config)

        self.do_reshape = reshape

        if n_channels is None:
            n_channels = self.pretrained_model.encoder.ch

        self.proj_norm = Normalize(in_channels,num_groups=in_channels//2)
        self.proj = nn.Conv2d(in_channels,n_channels,kernel_size=3,
                            stride=1,padding=1)

        blocks = []
        downs = []
        ch_in = n_channels
        for m in ch_mult:
            blocks.append(ResnetBlock(in_channels=ch_in,out_channels=m*n_channels,dropout=dropout))
            ch_in = m * n_channels
            downs.append(Downsample(ch_in, with_conv=False))

        self.model = nn.ModuleList(blocks)
        self.downsampler = nn.ModuleList(downs)


    def instantiate_pretrained(self, config):
        model = instantiate_from_config(config)
        self.pretrained_model = model.eval()
        # self.pretrained_model.train = False
        for param in self.pretrained_model.parameters():
            param.requires_grad = False


    @torch.no_grad()
    def encode_with_pretrained(self,x):
        c = self.pretrained_model.encode(x)
        if isinstance(c, DiagonalGaussianDistribution):
            c = c.mode()
        return  c

    def forward(self,x):
        z_fs = self.encode_with_pretrained(x)
        z = self.proj_norm(z_fs)
        z = self.proj(z)
        z = nonlinearity(z)

        for submodel, downmodel in zip(self.model,self.downsampler):
            z = submodel(z,temb=None)
            z = downmodel(z)

        if self.do_reshape:
            z = rearrange(z,'b c h w -> b (h w) c')
        return z

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3, 4])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3, 4])

    def nll(self, sample, dims=[1, 2, 3, 4]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class MultiScaleGrid(nn.Module):
    def __init__(self, in_dim, embed_dim=4, grid_size=[64, 32, 16, 8],
                        xyz_min=[-1, -1, -1], xyz_max = [1, 1, 1]):
        super().__init__()
        self.multi_embed = nn.ModuleList([])
        scale_levels = len(grid_size)
        for i in range(scale_levels):
            self.multi_embed.append(nn.Sequential(
                nn.Conv3d(in_dim, embed_dim, 3, 1, 1),
            )
                                    )
        self.grid_size = grid_size
        # self.xyz_min = torch.Tensor(xyz_min)
        # self.xyz_max = torch.Tensor(xyz_max)
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))

    def forward(self, grid, xyz):
        # grid: C, X, Y, Z             xyz: B, 3
        channels = grid.shape[0]
        grid = grid.unsqueeze(0)
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        # print(ind_norm.shape)
        tmp = F.grid_sample(grid, ind_norm, mode='bilinear', align_corners=True)
        # print(out.shape)
        tmp = tmp.reshape(channels, -1).T.reshape(*shape, channels)
        out = [tmp]
        for i, size in enumerate(self.grid_size):
            xs = F.interpolate(grid, size=(size, size, size), mode='trilinear')
            xs = self.multi_embed[i](xs)
            tmp = F.grid_sample(xs, ind_norm, mode='bilinear', align_corners=True)
            tmp = tmp.reshape(channels, -1).T.reshape(*shape, channels)
            out.append(tmp)
        return torch.cat(out, dim=-1)

class MultiScaleAttentionGrid(nn.Module):
    def __init__(self, in_dim, embed_dim=4, grid_size=[64, 32, 16, 8],
                        xyz_min=[-1, -1, -1], xyz_max = [1, 1, 1]):
        super().__init__()
        # self.multi_embed = nn.ModuleList([])
        scale_levels = len(grid_size)
        # for i in range(scale_levels):
        #     self.multi_embed.append(nn.Sequential(
        #         nn.Conv3d(in_dim, embed_dim, 3, 1, 1),
        #     )
        #                             )
        self.att = MSWA_3D(dim=in_dim, heads=1,
                                    window_size_q=[8, 8, 8],
                                    window_size_k=[[8, 8, 8], [4, 4, 4], [2, 2, 2]],
                                    groups=1)
        nn.init.normal_(self.att.out_conv[0].weight, 0, 0.001)
        nn.init.constant_(self.att.out_conv[0].bias, 0)
        self.grid_size = grid_size
        # self.xyz_min = torch.Tensor(xyz_min)
        # self.xyz_max = torch.Tensor(xyz_max)
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))

    def forward(self, grid, xyz):
        # grid: C, X, Y, Z             xyz: B, 3
        channels = grid.shape[0]
        grid = grid.unsqueeze(0)
        grid = self.att(grid)
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        # print(ind_norm.shape)
        tmp = F.grid_sample(grid, ind_norm, mode='bilinear', align_corners=True)
        # print(out.shape)
        tmp = tmp.reshape(channels, -1).T.reshape(*shape, channels)
        out = [tmp]
        for i, size in enumerate(self.grid_size):
            xs = F.interpolate(grid, size=(size, size, size), mode='trilinear')
            # xs = self.multi_embed[i](xs)
            tmp = F.grid_sample(xs, ind_norm, mode='bilinear', align_corners=True)
            tmp = tmp.reshape(channels, -1).T.reshape(*shape, channels)
            out.append(tmp)
        return torch.cat(out, dim=-1)

class AutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 classes=10,
                 # rgbnet_dim=4,
                 # rgbnet_width=256,
                 # rgbnet_depth=8,
                 cfg={}, **kwargs
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.down_ratio = 2 ** (len(ddconfig['ch_mult']) - 1)
        self.loss = LPIPSWithDiscriminator(**lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv3d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.cfg = cfg
        self.std_scale = cfg.get('std_scale', 1.)
        self.use_render_loss = cfg.get('use_render_loss', False)
        self.use_cls_loss = cfg.get('use_cls_loss', False)

        # model kwargs #
        self.register_buffer('xyz_min', torch.Tensor(self.cfg.render_kwargs.dvgo.xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(self.cfg.render_kwargs.dvgo.xyz_max))
        self.fast_color_thres = self.cfg.render_kwargs.dvgo.fast_color_thres
        self.mask_cache_thres = self.cfg.render_kwargs.dvgo.mask_cache_thres

        # determine based grid resolution
        self.num_voxels_base = self.cfg.render_kwargs.dvgo.num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1 / 3)

        # determine the density bias shift
        self.alpha_init = self.cfg.render_kwargs.dvgo.alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1 / (1 - self.alpha_init) - 1)]))
        self.act_shift -= 4
        self.num_voxels = self.cfg.render_kwargs.dvgo.num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        viewbase_pe = self.cfg.render_kwargs.dvgo.viewbase_pe
        self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(viewbase_pe)]))

        #self.residual = MultiScaleAttentionGrid(embed_dim - 1, grid_size=cfg.grid_size)
        dim0 = (3 + 3 * viewbase_pe * 2)
        if self.cfg.render_kwargs.dvgo.rgbnet_full_implicit:
            pass
        elif self.cfg.render_kwargs.dvgo.rgbnet_direct:
            dim0 += self.cfg.render_kwargs.dvgo.rgbnet_dim# * (1 + len(self.residual.grid_size))
        else:
            dim0 += self.cfg.render_kwargs.dvgo.rgbnet_dim - 3
        rgbnet_width = cfg.render_kwargs.dvgo.rgbnet_width
        rgbnet_depth = cfg.render_kwargs.dvgo.rgbnet_depth

        self.rgbnet = nn.Sequential(
            nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                for _ in range(rgbnet_depth - 2)
            ],
            nn.Linear(rgbnet_width, 3),
        )
        nn.init.constant_(self.rgbnet[-1].bias, 0)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list(), use_ema=False):
        sd = torch.load(path, map_location="cpu")
        sd_keys = sd.keys()
        if 'ema' in list(sd.keys()) and use_ema:
            sd = sd['ema']
            new_sd = {}
            for k in sd.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]    # remove ema_model.
                    new_sd[new_k] = sd[k]
            sd = new_sd
        else:
            if 'model' in sd_keys:
                sd = sd["model"]
            elif 'state_dict' in sd_keys:
                sd = sd['state_dict']
            else:
                raise ValueError("")
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        #### add key in model but not in sd ####
        self_state = self.state_dict()
        for k in list(self_state.keys()):
            if k not in sd:
                sd[k] = self_state[k]
        msg = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        print('==>Load AutoEncoder Info: ', msg)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def class_loss(self, field, classes_id):
        b = field.shape[0]
        # field = self.classifier_conv(field)
        # logits = self.classifier(field.view(b,-1))
        logits = self.classifier(field)
        loss_class = F.cross_entropy(logits, classes_id) * self.cfg.render_kwargs.weight_cls
        acc = (torch.max(logits, 1)[1].view(classes_id.size()).data == classes_id.data).sum() / b
        return loss_class, loss_class.detach().item(), acc.item()*100

    def total_variation(self, v, mask=None):
        tv2 = v.diff(dim=-3).abs()
        tv3 = v.diff(dim=-2).abs()
        tv4 = v.diff(dim=-1).abs()
        if mask is not None:
            tv2 = tv2[mask[:, :, :-1] & mask[:, :, 1:]]
            tv3 = tv3[mask[:, :, :, :-1] & mask[:, :, :, 1:]]
            tv4 = tv4[mask[:, :, :, :, :-1] & mask[:, :, :, :, 1:]]
        # return (tv2.mean() + tv3.mean() + tv4.mean()) / 3
        return (tv2.sum() + tv3.sum() + tv4.sum()) / v.shape[-1]

    def render_loss(self, field, render_kwargs, **kwargs):
        # assert len(render_kwargs) == len(field);
        HWs, Kss, nears, fars, i_trains, i_vals, i_tests, posess, imagess = [
            render_kwargs[k] for k in [
                'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'images'
            ]
        ]
        densities = field[:, 0].contiguous()
        features = field[:, 1:].contiguous()
        # features = features + self.residual(features)
        loss_total = 0.
        loss = 0.
        count = 0
        psnr = 0.
        class_ids = kwargs['class_id']
        accelerator = kwargs['accelerator']
        opt_nerf = kwargs['opt_nerf']

        for idx, (dens, fea, HW, Ks, near, far, i_train, i_val, i_test, poses, images, cls_id) in \
            enumerate(zip(densities, features, HWs, Kss, nears, fars, i_trains, i_vals, i_tests, posess, imagess, class_ids)):
        # for fie, render_kwarg in zip(field, render_kwargs):
            # HW, Ks, near, far, i_train, i_val, i_test, poses, images = [
            #     render_kwarg[k] for k in [
            #         'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'images'
            #     ]
            # ]
            device = dens.device
            rgb_tr_ori = images.to(device)
            render_kwarg_train = {
                'near': near,
                'far': far,
                'bg': self.cfg.render_kwargs.bg,
                'rand_bkgd': False,
                'stepsize': self.cfg.render_kwargs.stepsize,
                'inverse_y': self.cfg.render_kwargs.inverse_y,
                'flip_x': self.cfg.render_kwargs.flip_x,
                'flip_y': self.cfg.render_kwargs.flip_y,
                'class_id': cls_id
            }
            # in maskcache
            # rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = self.get_training_rays_in_maskcache_sampling(
            #     rgb_tr_ori=rgb_tr_ori,
            #     train_poses=poses,
            #     HW=HW, Ks=Ks,
            #     ndc=False, inverse_y=render_kwarg_train['inverse_y'],
            #     flip_x=render_kwarg_train['flip_x'], flip_y=render_kwarg_train['flip_y'],
            #     density=dens,
            #     render_kwargs=render_kwarg_train)
            # flatten
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = self.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses,
                HW=HW, Ks=Ks,
                ndc=False, inverse_y=render_kwarg_train['inverse_y'],
                flip_x=render_kwarg_train['flip_x'], flip_y=render_kwarg_train['flip_y'],
                )
            # render_kwarg.update()
            # with tqdm(initial=1, total=self.cfg.render_kwargs.inner_iter,
            #       disable=not accelerator.is_main_process) as pbar2:
            #     for iter in range(self.cfg.render_kwargs.inner_iter):
                # loss = 0.
            # print(rgb_tr.shape[0])
            with tqdm(initial=1, total=self.cfg.render_kwargs.inner_iter,
                      disable=not accelerator.is_main_process) as pbar2:
                for iter in range(self.cfg.render_kwargs.inner_iter):
                    # loss = 0.
                    sel_b = torch.randint(rgb_tr.shape[0], [self.cfg.render_kwargs.N_rand])
                    # sel_r = torch.randint(rgb_tr.shape[1], [self.cfg.render_kwargs.N_rand])
                    # sel_c = torch.randint(rgb_tr.shape[2], [self.cfg.render_kwargs.N_rand])
                    # target = rgb_tr[sel_b, sel_r, sel_c].to(device)
                    # rays_o = rays_o_tr[sel_b, sel_r, sel_c].to(device)
                    # rays_d = rays_d_tr[sel_b, sel_r, sel_c].to(device)
                    # viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
                    target = rgb_tr[sel_b].to(device)
                    rays_o = rays_o_tr[sel_b].to(device)
                    rays_d = rays_d_tr[sel_b].to(device)
                    viewdirs = viewdirs_tr[sel_b]
                    # fea2 = fea + self.residual(fea.unsqueeze(0))[0]
                    render_result = self.render_train(dens, fea, rays_o, rays_d, viewdirs, **render_kwarg_train)
                    loss_main = self.cfg.render_kwargs.weight_main * F.mse_loss(render_result['rgb_marched'], target)
                    psnr_cur = -10. * torch.log10(loss_main.detach()/self.cfg.render_kwargs.weight_main)
                    psnr += psnr_cur
                    pout = render_result['alphainv_last'].clamp(1e-6, 1 - 1e-6)
                    entropy_last_loss = -(pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)).mean()
                    loss_entropy_last = self.cfg.render_kwargs.weight_entropy_last * entropy_last_loss
                    rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
                    rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
                    loss_rgbper = self.cfg.render_kwargs.weight_rgbper * rgbper_loss
                    loss = loss_main + loss_entropy_last + loss_rgbper
                    opt_nerf.zero_grad()
                    accelerator.backward(loss)
                    opt_nerf.step()
                    pbar2.set_postfix({
                        'inner_iter': iter,
                        'render_loss': loss,
                        'psnr': psnr_cur,
                    })
            count += 1
            # opt.zero_grad()

            # opt.step()
            # pbar2.set_postfix({
            #     'inner_iter': iter,
            #     'render_loss': loss,
            #     'psnr': psnr_cur,
            # })
            # pbar2.update(1)
            loss_total += loss.detach().item()

            torch.cuda.empty_cache()
        return loss / count, loss_total / count, psnr.item() / count / self.cfg.render_kwargs.inner_iter

    @torch.no_grad()
    def get_training_rays_in_maskcache_sampling(self, rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, density,
                                                render_kwargs):
        # print('get_training_rays_in_maskcache_sampling: start')
        assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
        CHUNK = 64
        DEVICE = rgb_tr_ori[0].device
        eps_time = time.time()
        N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
        rgb_tr = torch.zeros([N, 3], device=DEVICE)
        rays_o_tr = torch.zeros_like(rgb_tr)
        rays_d_tr = torch.zeros_like(rgb_tr)
        viewdirs_tr = torch.zeros_like(rgb_tr)
        imsz = []
        top = 0
        for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
            assert img.shape[:2] == (H, W)
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
            mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)
            for i in range(0, img.shape[0], CHUNK):
                mask[i:i + CHUNK] = self.hit_coarse_geo(
                    rays_o=rays_o[i:i + CHUNK], rays_d=rays_d[i:i + CHUNK], density=density, **render_kwargs).to(DEVICE)
            n = mask.sum()
            rgb_tr[top:top + n].copy_(img[mask])
            rays_o_tr[top:top + n].copy_(rays_o[mask].to(DEVICE))
            rays_d_tr[top:top + n].copy_(rays_d[mask].to(DEVICE))
            viewdirs_tr[top:top + n].copy_(viewdirs[mask].to(DEVICE))
            imsz.append(n)
            top += n

        # print('get_training_rays_in_maskcache_sampling: ratio', top / N)
        rgb_tr = rgb_tr[:top]
        rays_o_tr = rays_o_tr[:top]
        rays_d_tr = rays_d_tr[:top]
        viewdirs_tr = viewdirs_tr[:top]
        eps_time = time.time() - eps_time
        # print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz

    @torch.no_grad()
    def get_training_rays_flatten(self, rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
        # print('get_training_rays_flatten: start')
        assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
        # eps_time = time.time()
        DEVICE = rgb_tr_ori[0].device
        N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
        rgb_tr = torch.zeros([N, 3], device=DEVICE)
        rays_o_tr = torch.zeros_like(rgb_tr)
        rays_d_tr = torch.zeros_like(rgb_tr)
        viewdirs_tr = torch.zeros_like(rgb_tr)
        imsz = []
        top = 0
        for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
            assert img.shape[:2] == (H, W)
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
            n = H * W
            rgb_tr[top:top + n].copy_(img.flatten(0, 1))
            rays_o_tr[top:top + n].copy_(rays_o.flatten(0, 1).to(DEVICE))
            rays_d_tr[top:top + n].copy_(rays_d.flatten(0, 1).to(DEVICE))
            viewdirs_tr[top:top + n].copy_(viewdirs.flatten(0, 1).to(DEVICE))
            imsz.append(n)
            top += n

        assert top == N
        # eps_time = time.time() - eps_time
        # print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz

    def hit_coarse_geo(self, rays_o, rays_d, density, near, stepsize, **render_kwargs):
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        mask_cache = self.forward_mask(density, ray_pts[mask_inbbox])
        hit[ray_id[mask_inbbox][mask_cache]] = 1
        return hit.reshape(shape)

    def forward_mask(self, density, xyz):
        # density: X, Y, Z
        dens = density.unsqueeze(0).unsqueeze(0)
        dens = F.max_pool3d(dens, kernel_size=3, padding=1, stride=1)
        alpha = 1 - torch.exp(
            -F.softplus(dens + self.act_shift * self.voxel_size_ratio))
        mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
        xyz_len = self.xyz_max - self.xyz_min
        xyz2ijk_scale = (torch.Tensor(list(mask.shape)).to(dens.device) - 1) / xyz_len
        xyz2ijk_shift = -self.xyz_min * xyz2ijk_scale
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask_cache = render_utils_cuda.maskcache_lookup(mask, xyz, xyz2ijk_scale, xyz2ijk_shift)
        mask_cache = mask_cache.reshape(shape)
        return mask_cache

    def forward_grid(self, grid, xyz):
        # grid: C, X, Y, Z
        channels = grid.shape[0]
        grid = grid.unsqueeze(0)
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        # print(ind_norm.shape)
        out = F.grid_sample(grid, ind_norm, mode='bilinear', align_corners=True)
        # print(out.shape)
        out = out.reshape(channels, -1).T.reshape(*shape, channels)
        if channels == 1:
            out = out.squeeze(-1)
        return out

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return dvgo.Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def render_train(self, dens, fea, rays_o, rays_d, viewdirs, **render_kwargs):
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, 'Only suuport point queries in [N, 3] format'
        # for fie in field:
        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        density = dens
        # skip known free space
        mask = self.forward_mask(density, ray_pts)
        # if self.mask_cache is not None:
        # mask = self.mask_cache(ray_pts)
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]
        step_id = step_id[mask]

        ### flip
        # self.density.grid.data = torch.flip(self.density.grid.data, dims=[-3, -2, -1])
        # self.k0.grid.data = torch.flip(self.k0.grid.data, dims=[-3, -2, -1])
        # query for alpha w/ post-activation
        density = self.forward_grid(density.unsqueeze(0), ray_pts)
        # density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = dvgo.Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for color
        # if self.rgbnet_full_implicit:
        #     pass
        # else:
        #     k0 = self.k0(ray_pts)
        k0 = self.forward_grid(fea, ray_pts)
        #k0 = self.residual(fea, ray_pts)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            if self.cfg.render_kwargs.dvgo.rgbnet_direct:
                k0_view = k0
            else:
                k0_view = k0[:, 3:]
                k0_diffuse = k0[:, :3]
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0, -2)[ray_id]
            rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
            # rgb_logit = self.rgbnet[render_kwargs['class_id']](rgb_feat)
            rgb_logit = self.rgbnet(rgb_feat)
            if self.cfg.render_kwargs.dvgo.rgbnet_direct:
                rgb = torch.sigmoid(rgb_logit)
            else:
                rgb = torch.sigmoid(rgb_logit + k0_diffuse)

        # Ray marching
        rgb_marched = segment_coo(
            src=(weights.unsqueeze(-1) * rgb),
            index=ray_id,
            out=torch.zeros([N, 3], device=weights.device),
            reduce='sum')
        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                    src=(weights * step_id),
                    index=ray_id,
                    out=torch.zeros([N], device=weights.device),
                    reduce='sum')
            ret_dict.update({'depth': depth})

        return ret_dict

    def training_step(self, inputs, optimizer_idx, global_step, **kwargs):
        # inputs = self.get_input(batch, self.image_key)
        # opt_nerf = kwargs['opt_nerf']
        render_kwargs = inputs["render_kwargs"]
        class_id = inputs["class_id"]
        inputs = inputs['input'] / self.std_scale
        reconstructions, posterior = self(inputs)
        if 'accelerator' in kwargs:
            accelerator = kwargs['accelerator']
        else:
            accelerator = None
        if 'opt_nerf' in kwargs:
            opt_nerf = kwargs['opt_nerf']
        else:
            opt_nerf = None

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, global_step,
                                            last_layer=self.get_last_layer(), split="train")
            # self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            if self.cfg.render_kwargs.weight_tv > 0:
                # grid = field[idx].unsqueeze(0)
                # wx = wy = wz = self.cfg.render_kwargs.weight_tv / len(rays_o)
                # total_variation_cuda.total_variation_add_grad(
                #     grid, grid.grad, wx, wy, wz, False)
                grid = reconstructions
                loss_tv = self.total_variation(grid) * self.cfg.render_kwargs.weight_tv
                aeloss += loss_tv
            if self.use_cls_loss and global_step >= self.cfg.get('cls_start', 0):
                cls_loss, cls_loss_item, acc = self.class_loss(reconstructions, class_id)
                aeloss += cls_loss
                log_dict_ae.update({'cls_loss': cls_loss_item, 'acc': acc})
            if self.use_render_loss and global_step >= self.cfg.render_start:
                # print(self.cfg.render_start)
                render_loss, render_loss_item, psnr = self.render_loss(
                    # max_min_unnormalize(reconstructions * self.std_scale, self.cfg.maxm, self.cfg.minm),
                    reconstructions.detach() * self.std_scale,
                    render_kwargs, class_id=class_id, accelerator=accelerator, opt_nerf=opt_nerf)
                # aeloss += render_loss
                log_dict_ae.update({'render_loss': render_loss_item, 'psnr': psnr})
            return aeloss, log_dict_ae

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, global_step,
                                                last_layer=self.get_last_layer(), split="train")

            # self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            # self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss, log_dict_disc

    def render_step(self, inputs, accelerator, opt):
        render_kwargs = inputs["render_kwargs"]
        inputs = inputs['input'] / self.std_scale
        reconstructions, posterior = self(inputs)
        reconstructions = reconstructions
        render_loss, render_loss_item, psnr = self.render_loss(reconstructions * self.std_scale, render_kwargs,
                                                               accelerator, opt)
        return render_loss, {'render_loss': render_loss_item, 'psnr': psnr}

    def validation_step(self, inputs, global_step):
        # inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, global_step,
                                            last_layer=self.get_last_layer(), split="val")

        # self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        # self.log_dict(log_dict_ae)
        # self.log_dict(log_dict_disc)
        return log_dict_ae, log_dict_disc

    def validate_img(self, inputs):
        inputs = inputs['input']
        reconstructions, posterior = self(inputs)
        return reconstructions

    # def configure_optimizers(self):
    #     lr = self.learning_rate
    #     opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
    #                               list(self.decoder.parameters())+
    #                               list(self.quant_conv.parameters())+
    #                               list(self.post_quant_conv.parameters()),
    #                               lr=lr, betas=(0.5, 0.9))
    #     opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
    #                                 lr=lr, betas=(0.5, 0.9))
    #     return [opt_ae, opt_disc], []

    @torch.no_grad()
    def render_img(self, inputs, render_kwargs):
        input = inputs['input']
        rotate_flag = render_kwargs.rotate_flag
        if rotate_flag:
            angle, axes = inputs['rotate_params']
        device = input.device
        H, W, focal = render_kwargs.hwf
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])
        try:
            render_poses = inputs['render_kwargs']['poses']
        except:
            render_pose = torch.stack([pose_spherical(angle, -30.0, 3.0) for angle in np.linspace(-180,180,5)[:-1]], 0)
            render_poses = [render_pose for _ in range(input.shape[0])]

        # Ks = render_kwargs['Ks']
        ndc = render_kwargs.ndc
        render_factor = render_kwargs.render_factor
        if render_factor != 0:
            # HW = np.copy(HW)
            # Ks = np.copy(Ks)
            H = int(H / render_factor)
            W = int(W / render_factor)
            K[:2, :3] /= render_factor

        #### model ####
        reconstructions, posterior = self(input / self.std_scale)
        # cls_logits = self.classifier_conv(reconstructions)
        # cls_logits = self.classifier(cls_logits.view(reconstructions.shape[0], -1))
        # cls_logits = self.classifier(reconstructions)
        # cls_ids = torch.max(cls_logits, 1)[1]
        # reconstructions = max_min_unnormalize(reconstructions * self.std_scale, self.cfg.maxm, self.cfg.minm)
        reconstructions = reconstructions * self.std_scale
        # reconstructions.clamp_(-1, 1)
        # reconstructions = reconstructions / (1 - reconstructions.abs())
        if rotate_flag:
            reconstructions = self.inv_rotate(reconstructions.detach().cpu().numpy(), angle, axes).to(device)
        # reconstructions = input
        rgbs = []
        depths = []
        bgmaps = []
        for idx_obj in range(reconstructions.shape[0]):

            # rgbs = []
            # depths = []
            # bgmaps = []
            dens = reconstructions[idx_obj][0]
            fea = reconstructions[idx_obj][1:]
            # fea = fea + self.residual(fea.unsqueeze(0))[0]
            render_pose = render_poses[idx_obj]

            # render_kwargs['class_id'] = cls_ids[idx_obj].item()
            for i, c2w in enumerate(render_pose):
                # H, W = HW[i]
                # K = Ks[i]
                # H, W = HW
                # K = Ks
                if not isinstance(c2w, torch.Tensor):
                    c2w = torch.Tensor(c2w)
                rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, ndc, inverse_y=render_kwargs.inverse_y,
                    flip_x=render_kwargs.flip_x, flip_y=render_kwargs.flip_y)
                keys = ['rgb_marched', 'depth', 'alphainv_last']
                rays_o = rays_o.flatten(0, -2).to(device)
                rays_d = rays_d.flatten(0, -2).to(device)
                viewdirs = viewdirs.flatten(0, -2).to(device)
                render_result_chunks = [
                    {k: v for k, v in self.render_train(dens, fea, ro, rd, vd, **render_kwargs).items() if k in keys}
                    for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
                ]
                render_result = {
                    k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H, W, -1)
                    for k in render_result_chunks[0].keys()
                }

                if render_kwargs.render_depth:
                    depth = render_result['depth'].cpu().numpy()
                    depths.append(depth)
                rgb = render_result['rgb_marched'].cpu().numpy()
                bgmap = render_result['alphainv_last'].cpu().numpy()
                rgbs.append(rgb)
                bgmaps.append(bgmap)
            # rgbs = np.array(rgbs)
            # depths = np.array(depths)
            # bgmaps = np.array(bgmaps)
        rgbs = np.array(rgbs)
        depths = np.array(depths)
        bgmaps = np.array(bgmaps)
        # del model
        torch.cuda.empty_cache()
        return rgbs, depths, bgmaps

    def render_img2(self, inputs, render_kwargs): ### use previous rgbnet weight
        input = inputs['input']
        rotate_flag = render_kwargs.rotate_flag
        if rotate_flag:
            angle, axes = inputs['rotate_params']
        device = input.device
        # print('angle:', angle)
        # print('axes:', axes)
        # obj_kwargs = inputs['kwargs']
        rgb_net_weight = inputs['rgb_net_weight']
        # xyz_min = render_kwargs['xyz_min']
        # xyz_max = render_kwargs['xyz_max']
        # num_voxels = render_kwargs['num_voxels']
        # num_voxels_base = render_kwargs['num_voxels_base']
        # alpha_init = render_kwargs['alpha_init']
        # fast_color_thres = render_kwargs['fast_color_thres']
        # rgbnet_dim = render_kwargs['rgbnet_dim']
        # rgbnet_direct = render_kwargs['rgbnet_direct']
        H, W, focal = render_kwargs.hwf
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])
        try:
            render_poses = render_kwargs.poses
        except:
            render_poses = torch.stack([pose_spherical(angle, -30.0, 3.0) for angle in np.linspace(-180,180,5)[:-1]], 0)
        # Ks = render_kwargs['Ks']
        ndc = render_kwargs.ndc
        render_factor = render_kwargs.render_factor
        if render_factor != 0:
            # HW = np.copy(HW)
            # Ks = np.copy(Ks)
            H = (H / render_factor).astype(int)
            W = (W / render_factor).astype(int)
            K[:2, :3] /= render_factor

        #### model ####
        reconstructions, posterior = self(input / self.std_scale)
        reconstructions = reconstructions * self.std_scale
        # reconstructions.clamp_(-1, 1)
        # reconstructions = reconstructions / (1 - reconstructions.abs())
        if rotate_flag:
            reconstructions = self.inv_rotate(reconstructions.detach().cpu().numpy(), angle, axes).to(device)
            # input = self.inv_rotate(input.detach().cpu().numpy(), angle, axes).to(device)

        # input[input<-1] = 0
        model_kwargs = render_kwargs.dvgo
        model = dvgo.DirectVoxGO(**model_kwargs)
        # msg = model.load_state_dict(rgb_net_weight, strict=False)
        model.act_shift -= 4
        model = model.to(device)
        model.eval()
        rgbs = []
        depths = []
        bgmaps = []
        for idx_obj in range(reconstructions.shape[0]):
            msg = model.load_state_dict(rgb_net_weight[idx_obj], strict=False)
            model.density.grid.data = reconstructions[idx_obj, 0].expand(1, 1, *(reconstructions[idx_obj, 0].shape))
            model.k0.grid.data = reconstructions[idx_obj, 1:].unsqueeze(0)
            # model.density.grid.data = input[idx_obj, 0].expand(1, 1, *(input[idx_obj, 0].shape))
            # model.k0.grid.data = input[idx_obj, 1:].unsqueeze(0)
            if 'mask_cache' in inputs.keys():
                model.mask_cache.mask = inputs['mask_cache'][idx_obj]
            # rgbs = []
            # depths = []
            # bgmaps = []
            for i, c2w in enumerate(render_poses):
                # H, W = HW[i]
                # K = Ks[i]
                # H, W = HW
                # K = Ks
                c2w = torch.Tensor(c2w)
                rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, ndc, inverse_y=render_kwargs.inverse_y,
                    flip_x=render_kwargs.flip_x, flip_y=render_kwargs.flip_y)
                keys = ['rgb_marched', 'depth', 'alphainv_last']
                rays_o = rays_o.flatten(0, -2).to(device)
                rays_d = rays_d.flatten(0, -2).to(device)
                viewdirs = viewdirs.flatten(0, -2).to(device)
                render_result_chunks = [
                    {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
                    for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
                ]
                render_result = {
                    k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H, W, -1)
                    for k in render_result_chunks[0].keys()
                }

                if render_kwargs.render_depth:
                    depth = render_result['depth'].cpu().numpy()
                    depths.append(depth)
                rgb = render_result['rgb_marched'].cpu().numpy()
                bgmap = render_result['alphainv_last'].cpu().numpy()
                rgbs.append(rgb)
                bgmaps.append(bgmap)
            # rgbs = np.array(rgbs)
            # depths = np.array(depths)
            # bgmaps = np.array(bgmaps)
        rgbs = np.array(rgbs)
        depths = np.array(depths)
        bgmaps = np.array(bgmaps)
        del model
        torch.cuda.empty_cache()
        return rgbs, depths, bgmaps

    def inv_rotate(self, image, angle, axes):
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        res = []
        for i in range(image.shape[0]):  # batch size
            single_image = image[i]
            for j in range(single_image.shape[0]):  # channel
                single_image[j] = ndimage.rotate(single_image[j], angle=-angle[i], axes=axes[i], reshape=False)
            res.append(torch.from_numpy(single_image))
        res = torch.stack(res, dim=0)
        return res

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    '''
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log
    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
    '''

if __name__ == '__main__':
    # ddconfig = {'double_z': True,
    #   'z_channels': 4,
    #   'resolution': (64, 64, 64),
    #   'in_channels': 4,
    #   'out_ch': 4,
    #   'ch': 32,
    #   'groups': 8,
    #   'ch_mult': [ 1,2,2,2 ],  # num_down = len(ch_mult)-1
    #   'num_res_blocks': 2,
    #   'attn_resolutions': [ ],
    #   'dropout': 0.0}
    # lossconfig = {'disc_start': 50001,
    #     'kl_weight': 0.000001,
    #     'disc_weight': 0.5,
    #     'perceptual_weight': 0,
    #     }
    # # model = AutoencoderKL(ddconfig, lossconfig, embed_dim=4,
    # #                       ckpt_path='/pretrain_weights/model-kl-f8.ckpt', )
    # model = AutoencoderKL(ddconfig, lossconfig, embed_dim=4)
    '''
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=0.01)
    lr_lambda = lambda iter: (1 - iter / 1000) ** 0.95
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    for s in range(1000):
        lr_scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        print(cur_lr)
    '''
    # x = torch.rand(1, 4, 128, 128, 128)
    # with torch.no_grad():
    #     y = model(x)
    pass
