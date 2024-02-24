import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import torch.distributions.categorical as cate
from torch_scatter import segment_coo, segment_csr
from diff_nerf import dvgo, grid
from functools import partial
from torch.utils.cpp_extension import load
import pdb
import time
#from diff_nerf.mesh_utils import (
#    MeshGenerator,
#    export_meshes_to_path,
#    reconstruct_meshes_from_model,
#)
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

####   DVGO NeRF  ####
# Compared to nerf_module: add part shape and part texture features
# loss with images not rays

class NeRF(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(NeRF, self).__init__()
        # print(cfg)
        self.cfg = cfg
        self.register_buffer('xyz_min', torch.Tensor(self.cfg.dvgo.xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(self.cfg.dvgo.xyz_max))
        self.fast_color_thres = self.cfg.dvgo.fast_color_thres
        self.mask_cache_thres = self.cfg.dvgo.mask_cache_thres

        # determine based grid resolution
        self.num_voxels_base = self.cfg.dvgo.num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1 / 3)

        # determine the density bias shift
        self.alpha_init = self.cfg.dvgo.alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1 / (1 - self.alpha_init) - 1)]))
        self.act_shift -= 4
        self.num_voxels = self.cfg.dvgo.num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base

        viewbase_pe = self.cfg.dvgo.viewbase_pe
        self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(viewbase_pe)]))
        self.use_barf_pe = self.cfg.dvgo.get('use_barf_pe', False)
        # self.residual = MultiScaleAttentionGrid(embed_dim - 1, grid_size=cfg.grid_size)
        dim0 = (3 + 3 * viewbase_pe * 2)
        if self.cfg.dvgo.rgbnet_full_implicit:
            pass
        elif self.cfg.dvgo.rgbnet_direct:
            dim0 += self.cfg.dvgo.rgbnet_dim  # * (1 + len(self.residual.grid_size))
        else:
            dim0 += self.cfg.dvgo.rgbnet_dim - 3
        rgbnet_width = self.cfg.dvgo.rgbnet_width
        rgbnet_depth = self.cfg.dvgo.rgbnet_depth

        # part kwargs
        self.num_parts = self.cfg.get('num_parts', 4)
        self.part_fea_dim = self.cfg.get('part_fea_dim', 128)
        # self.part_embeddings = nn.Embedding(self.num_parts, self.part_fea_dim)
        # nn.init.normal_(self.part_embeddings.weight.data, 0.0, 1.0 / math.sqrt(self.part_fea_dim))
        # self.part_mlp = nn.Linear(self.part_fea_dim, self.part_fea_dim)
        # self.part_mlp = PartAtt(self.part_fea_dim, self.part_fea_dim, n_layers=4)
        # dim_index = 3 + 3 * viewbase_pe * 2
        self.index_mlp = IndexMLP(in_dim=self.cfg.dvgo.rgbnet_dim, out_dim=self.num_parts+1,
                                  part_dim=self.part_fea_dim, hidden_dim=self.part_fea_dim)

        # dim0 += self.part_fea_dim
        self.feat_mlp = RelateMLP(in_dim=dim0, out_dim=self.part_fea_dim, part_dim=self.part_fea_dim)
        # part render mlps
        self.rgbnets = nn.ModuleList()
        for _ in range(self.num_parts):
            rgbnet = nn.Sequential(
                nn.Linear(self.part_fea_dim, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth - 2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(rgbnet[-1].bias, 0)
            self.rgbnets.append(rgbnet)

    def forward(self, field, render_kwargs, **kwargs):
        return self.render_loss(field, render_kwargs, **kwargs)

    def render_loss(self, field, render_kwargs, **kwargs):
        # field =
        # assert len(render_kwargs) == len(field);
        global_step = kwargs.get('global_step', 0)
        HWs, Kss, nears, fars, i_trains, i_vals, i_tests, posess, imagess, maskss = [
            render_kwargs[k] for k in [
                'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'images', 'parts'
            ]
        ]
        loss = 0.
        # loss_item = 0.
        psnr = 0.
        bs = field.shape[0]
        densities = field[:, 0]
        features = field[:, 1:]
        accelerator = kwargs['accelerator']
        opt_nerf = kwargs['opt_nerf']
        joint_learn = kwargs.get('joint_learn', False)
        part_shape_feas = kwargs.get('part_shape_fea')
        part_text_feas = kwargs.get('part_texture_fea')
        # part_fea = kwargs.get('part_fea')
        # features = features + self.residual(features)
        loss_weights = kwargs['loss_weight'] if 'loss_weight' in kwargs else torch.ones(len(field),)
        for dens, fea, HW, Ks, near, far, i_train, i_val, i_test, poses, images, masks, lw, psf, ptf in \
                zip(densities, features, HWs, Kss, nears, fars, i_trains, i_vals, i_tests, posess, imagess, maskss, loss_weights,
                    part_shape_feas, part_text_feas):
            device = dens.device
            rgb_tr_ori = images.to(device)
            masks = masks.to(device)
            render_kwarg_train = {
                'near': self.cfg.near,
                'far': self.cfg.far,
                'bg': self.cfg.bg,
                'rand_bkgd': False,
                'stepsize': self.cfg.stepsize,
                'inverse_y': self.cfg.inverse_y,
                'flip_x': self.cfg.flip_x,
                'flip_y': self.cfg.flip_y,
                'render_mask': self.cfg.render_mask,
                'render_depth': self.cfg.render_depth
            }
            # loss = 0.
            if global_step >= self.cfg.get('maskcache_sampling_step'):
                rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, mask_tr = self.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    masks=masks,
                    train_poses=poses,
                    HW=HW, Ks=Ks,
                    ndc=False, inverse_y=False,
                    flip_x=False, flip_y=False,
                    density=dens,
                    render_kwargs=render_kwarg_train)
            else:
                rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, mask_tr = self.get_training_rays_flatten(
                    rgb_tr_ori=rgb_tr_ori,
                    masks=masks,
                    train_poses=poses,
                    HW=HW, Ks=Ks,
                    ndc=False, inverse_y=render_kwarg_train['inverse_y'],
                    flip_x=render_kwarg_train['flip_x'], flip_y=render_kwarg_train['flip_y'],
                )
            # with tqdm(initial=1, total=self.cfg.inner_iter,
            #           disable=not accelerator.is_main_process) as pbar2:
            if not joint_learn:
                inner_iter = self.cfg.inner_iter_indep
            else:
                inner_iter = self.cfg.inner_iter
            for iter in range(inner_iter):
                # loss = 0.
                # t_cur = time.time()
                sel_b = torch.randint(rgb_tr.shape[0], [self.cfg.N_rand])
                target = rgb_tr[sel_b].to(device)
                target_m = mask_tr[sel_b].to(device)
                rays_o = rays_o_tr[sel_b].to(device)
                rays_d = rays_d_tr[sel_b].to(device)
                viewdirs = viewdirs_tr[sel_b]

                # t_n = time.time()
                # print(t_n - t_cur)
                render_result = self.render_train(dens, fea, rays_o, rays_d, viewdirs,
                                                  psf, ptf, **render_kwarg_train)
                # t_n2 = time.time()
                # print(t_n2 - t_n)
                loss_main = self.cfg.weight_main * F.mse_loss(render_result['rgb_marched'], target)
                psnr_cur = -10. * torch.log10(loss_main.detach() / self.cfg.weight_main)
                # print(psnr_cur)
                psnr += psnr_cur
                pout = render_result['alphainv_last'].clamp(1e-6, 1 - 1e-6)
                entropy_last_loss = -(pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)).mean()
                loss_entropy_last = self.cfg.weight_entropy_last * entropy_last_loss
                rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
                rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
                loss_rgbper = self.cfg.weight_rgbper * rgbper_loss
                # loss_ind_entropy = self.cfg.weight_ind_en * self.loss_entropy(render_result['index'])
                # loss += loss_main + loss_entropy_last + loss_rgbper
                # torch.cuda.empty_cache()
                loss += (loss_main + loss_entropy_last + loss_rgbper) * lw.item()
                #loss_part_cons = self.cfg.weight_part_cons * self.loss_part_cons(render_result['feat'], render_result['index'])
                #loss += loss_part_cons * lw.item()
                if mask_tr is not None:
                    loss_mask = self.cfg.weight_mask * F.cross_entropy(render_result['part_marched'], target_m,
                                                                       ignore_index=0)
                    # loss_mask = 0.
                    # loss_mask_per = self.cfg.weight_mask_per * F.cross_entropy(render_result['index_pred_ray'],
                    #                            target_m[render_result['ind_uniques']], ignore_index=0)
                    # loss_comparable = self.cfg.weight_comparable * self.loss_comparable(
                    #                                 render_result['index'], render_result['ray_id'], target_m)
                    # loss_coverage = self.cfg.weight_coverage * self.loss_coverage(render_result['index_value'], target_m)
                    loss += loss_mask * lw.item() # + loss_coverage
                # loss = loss * lw.item()
                # t_n3 = time.time()
                # print(t_n3 - t_n2)
                if not joint_learn:
                    opt_nerf.zero_grad()
                    accelerator.backward(loss)
                    opt_nerf.step()
                    loss = 0.
                    # t_n4 = time.time()
                    # print(t_n4 - t_n3)
                # loss_item += loss.detach().item()
                # accelerator.backward(loss)
        # loss_item = loss.detach().item()
        loss_dict = {
            'loss_render_main': loss_main.detach().item(),
            'psnr': psnr/bs/inner_iter,
            # 'loss_ind_entropy': loss_ind_entropy.detach().item(),
            # 'loss_part_cons': loss_part_cons,
        }
        if mask_tr is not None:
            loss_dict['loss_mask'] = loss_mask.detach().item()
            # loss_dict['loss_mask_per'] = loss_mask_per.detach().item()
        #return loss/bs/self.cfg.inner_iter, loss_item/bs/self.cfg.inner_iter, psnr/bs/self.cfg.inner_iter
        return loss / bs / inner_iter, loss_dict

    def loss_coverage(self, pred, target):
        positive = target == 1
        positive_rays_implicit = pred[positive]
        loss = -torch.log(positive_rays_implicit + 1e-6)
        return loss.mean()

    def loss_comparable(self, index, ray_id, target):
        # index: N, num_parts
        num_part = []
        target = target[ray_id]
        positive = target == 1
        index = torch.max(index, dim=-1)[1]
        index = index[positive]
        for i in range(self.num_parts):
            temp = index == i
            # if temp.sum() > 0:
            num_part.append(temp.sum())
            # else:
            #     num_part.append(torch.Tensor(0., device=index.device))
        num_part = torch.stack(num_part, dim=0).float()
        loss = F.l1_loss(num_part, num_part.mean(dim=0, keepdim=True).repeat(num_part.shape[0]))
        return loss

    def loss_entropy(self, index):
        index = F.softmax(index, dim=-1)
        loss = cate.Categorical(probs=index).entropy() / math.log(index.shape[1])
        return loss.mean()

    def loss_part_cons(self, fea, index):
        # target = target[ray_id]
        # positive = target == 1
        index = torch.max(index, dim=-1)[1]
        # index = index[positive]
        # fea = fea[positive]
        # part_feas = []
        clusters = []
        loss = 0.
        for i in range(self.num_parts):
            temp = index == i
            if temp.sum() > 0:
                part_fea = fea[temp]
                # part_feas.append(part_fea)
                cluster = part_fea.mean(dim=0).unsqueeze(0)
                clusters.append(cluster)
                loss += (1 - F.cosine_similarity(part_fea, cluster.expand(part_fea.shape[0], -1))).mean()
        for i in range(len(clusters) - 1):
            clus_1 = clusters[i]
            for j in range(i+1, len(clusters)):
                clus_2 = clusters[j]
                loss += F.cosine_similarity(clus_1, clus_2).mean()
        return loss

    @torch.no_grad()
    def get_training_rays_in_maskcache_sampling(self, rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y,
                                                flip_x, flip_y, density, render_kwargs, masks=None):
        # print('get_training_rays_in_maskcache_sampling: start')
        assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and \
               len(rgb_tr_ori) == len(HW)
        CHUNK = 64
        DEVICE = rgb_tr_ori[0].device
        # eps_time = time.time()
        N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
        rgb_tr = torch.zeros([N, 3], device=DEVICE)
        if masks is not None:
            mask_tr = torch.zeros([N, ], dtype=torch.long, device=DEVICE)
        else:
            mask_tr = None
        rays_o_tr = torch.zeros_like(rgb_tr)
        rays_d_tr = torch.zeros_like(rgb_tr)
        viewdirs_tr = torch.zeros_like(rgb_tr)
        imsz = []
        top = 0
        if masks is not None:
            for c2w, img, (H, W), K, mas in zip(train_poses, rgb_tr_ori, HW, Ks, masks):
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
                p_f = mas[mask]
                part_label = torch.zeros(p_f.shape[0], dtype=torch.long, device=p_f.device)
                label_1_mask = (p_f[..., 0] < 20) & (p_f[..., 1] < 20) & (p_f[..., 2] >= 200)
                label_2_mask = (p_f[..., 0] >= 200) & (p_f[..., 1] < 20) & (p_f[..., 2] < 20)
                label_3_mask = (p_f[..., 0] < 20) & (p_f[..., 1] >= 200) & (p_f[..., 2] < 20)
                label_4_mask = (p_f[..., 0] >= 200) & (p_f[..., 1] >= 200) & (p_f[..., 2] < 20)
                part_label[label_1_mask] = 1
                part_label[label_2_mask] = 2
                part_label[label_3_mask] = 3
                part_label[label_4_mask] = 4
                mask_tr[top:top + n].copy_(part_label)
                rays_o_tr[top:top + n].copy_(rays_o[mask].to(DEVICE))
                rays_d_tr[top:top + n].copy_(rays_d[mask].to(DEVICE))
                viewdirs_tr[top:top + n].copy_(viewdirs[mask].to(DEVICE))
                imsz.append(n)
                top += n
        else:
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
        if masks is not None:
            mask_tr = mask_tr[:top]
        rgb_tr = rgb_tr[:top]
        rays_o_tr = rays_o_tr[:top]
        rays_d_tr = rays_d_tr[:top]
        viewdirs_tr = viewdirs_tr[:top]
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, mask_tr

    @torch.no_grad()
    def get_training_rays_flatten(self, rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y,
                                  flip_x, flip_y, masks=None):
        assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
        DEVICE = rgb_tr_ori[0].device
        N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
        rgb_tr = torch.zeros([N, 3], device=DEVICE)
        if masks is not None:
            mask_tr = torch.zeros([N, ], dtype=torch.long, device=DEVICE)
        else:
            mask_tr = None
        rays_o_tr = torch.zeros_like(rgb_tr)
        rays_d_tr = torch.zeros_like(rgb_tr)
        viewdirs_tr = torch.zeros_like(rgb_tr)
        imsz = []
        top = 0
        if masks is not None:
            for c2w, img, (H, W), K, mas in zip(train_poses, rgb_tr_ori, HW, Ks, masks):
                assert img.shape[:2] == (H, W)
                rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                    inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
                n = H * W
                rgb_tr[top:top + n].copy_(img.flatten(0, 1))
                p_f = mas.flatten(0, 1)
                part_label = torch.zeros(p_f.shape[0], dtype=torch.long, device=p_f.device)
                label_1_mask = (p_f[..., 0] < 20) & (p_f[..., 1] < 20) & (p_f[..., 2] >= 200)
                label_2_mask = (p_f[..., 0] >= 200) & (p_f[..., 1] < 20) & (p_f[..., 2] < 20)
                label_3_mask = (p_f[..., 0] < 20) & (p_f[..., 1] >= 200) & (p_f[..., 2] < 20)
                label_4_mask = (p_f[..., 0] >= 200) & (p_f[..., 1] >= 200) & (p_f[..., 2] < 20)
                part_label[label_1_mask] = 1
                part_label[label_2_mask] = 2
                part_label[label_3_mask] = 3
                part_label[label_4_mask] = 4
                mask_tr[top:top + n].copy_(part_label)
                # mask_tr[top:top + n].copy_(mas.flatten(0, 1))
                rays_o_tr[top:top + n].copy_(rays_o.flatten(0, 1).to(DEVICE))
                rays_d_tr[top:top + n].copy_(rays_d.flatten(0, 1).to(DEVICE))
                viewdirs_tr[top:top + n].copy_(viewdirs.flatten(0, 1).to(DEVICE))
                imsz.append(n)
                top += n
        else:
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
        #print(mask_tr.dtype)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, mask_tr

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
        #pdb.set_trace()
        dens = density.unsqueeze(0).unsqueeze(0)
        dens = F.max_pool3d(dens, kernel_size=3, padding=1, stride=1)
        alpha = 1 - torch.exp(
            -F.softplus(dens + self.act_shift * self.voxel_size_ratio))
        mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
        # print(mask.sum())
        xyz_len = self.xyz_max - self.xyz_min
        #pdb.set_trace()
        xyz2ijk_scale = (torch.Tensor(list(mask.shape)).to(dens.device) - 1) / xyz_len
        xyz2ijk_shift = -self.xyz_min * xyz2ijk_scale
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask_cache = render_utils_cuda.maskcache_lookup(mask, xyz, xyz2ijk_scale, xyz2ijk_shift)
        mask_cache = mask_cache.reshape(shape)
        #print(mask_cache.sum())
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

    def render_train(self, dens, fea, rays_o, rays_d, viewdirs, part_shape_fea, part_text_fea, **render_kwargs):
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, 'Only suuport point queries in [N, 3] format'
        # for fie in field:
        ret_dict = {}
        N = len(rays_o)
        # time1 = time.time()
        # sample points on rays
        ray_pts, ray_id, step_id = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        # print(ray_id)
        density = dens
        # print(density.max())
        # skip known free space
        mask = self.forward_mask(density, ray_pts)
        # print(mask.sum())
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
            #if mask.sum() < 1:
            #    mask = (alpha > -10.)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = dvgo.Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            #if mask.sum() < 1:
            #    mask = (weights > -10.)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]

        # query for color
        batch = ray_pts.shape[0]
        k0 = self.forward_grid(fea, ray_pts)
        # time2 = time.time()
        # print(time2 - time1)
        # rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        # xyz_emb = (rays_xyz.unsqueeze(-1) * self.viewfreq).flatten(-2)
        # xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)

        # part_fea = self.part_mlp(self.part_embeddings.weight)
        # index_pred = self.index_mlp(xyz_emb.unsqueeze(1), part_fea[None, ::].repeat(batch, 1, 1)).squeeze(1)
        index_pred = [self.index_mlp(in1, in2) for in1, in2 in zip(k0.unsqueeze(1).split(8192, 0), \
                                                       part_shape_fea[None, ::].expand(batch, -1, -1).split(8192, 0)
                                                    )]
        index_pred = torch.cat(index_pred, dim=0).squeeze(1) # B, num_parts+1
        # time0 = time.time()
        part_marched = segment_coo(
            src=(weights.unsqueeze(-1) * index_pred),
            index=ray_id,
            out=torch.zeros([N, self.num_parts + 1], device=weights.device),
            reduce='sum')
        # time1 = time.time()
        # print(time1 - time0)
        # max_points = segment_coo(
        #     src=(weights),
        #     index=ray_id,
        #     out=torch.zeros([N], device=weights.device),
        #     reduce='max')
        mlp_idxs = torch.max(part_marched, dim=-1)[1]
        # time3 = time.time()
        # print(time3 - time2)
        ind_uniques, ind_counts = torch.unique(ray_id, return_counts=True)
        # if ray_id.shape[0] > 0:
        #     index_pred_split = index_pred.split(list(ind_counts), 0)
        #     max_point_len = ind_counts.max()
        #     weights_ray_split = weights.split(list(ind_counts), 0)
        #     weights_ray_split = [F.pad(drs, pad=(0, max_point_len - drs.shape[0]), value=-100.) \
        #                          for drs in weights_ray_split]
        #     weights_ray_split = torch.stack(weights_ray_split, dim=0)  # ray_len, max_point_len
        #     max_point_id = torch.max(weights_ray_split, dim=1)[1]
        #
        #     # mlp_idxs = []
        #     # index_pred_rays = []
        #     index_pred_split = [F.pad(ips, pad=(0, 0, 0, max_point_len-ips.shape[0]), value=-100.) \
        #                         for ips in index_pred_split]
        #     index_pred_split = torch.stack(index_pred_split, dim=0) # ray_len, max_point_len, num_parts+1
        #     # index_pred_point = index_pred_split[max_point_id]
        #     index_gather = max_point_id[..., None, None].expand(-1, -1, index_pred_split.shape[-1])
        #     index_pred_point = torch.gather(index_pred_split, dim=1, index=index_gather).squeeze(1)
        # else:
        #     index_pred_point = None
        # time4 = time.time()
        # print(time4 - time3)
        if ray_id.shape[0] > 0:
            if self.rgbnets is None:
                # no view-depend effect
                rgb = torch.sigmoid(k0)
            else:
                # view-dependent color emission
                if self.cfg.dvgo.rgbnet_direct:
                    k0_view = k0
                else:
                    k0_view = k0[:, 3:]
                    k0_diffuse = k0[:, :3]
                viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
                viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
                viewdirs_emb = viewdirs_emb.flatten(0, -2)[ray_id]
                k0_view = torch.cat([k0_view, viewdirs_emb], -1)
                rgb_feat = [self.feat_mlp(in1, in2) for in1, in2 in zip(k0_view.unsqueeze(1).split(8192, 0), \
                                                           part_text_fea[None, ::].expand(batch, -1, -1).split(8192, 0)
                                                        )]
                rgb_feat = torch.cat(rgb_feat, dim=0).squeeze(1)
                # rgb_feat = torch.cat([rgb_feat, point_fea], dim=-1)
                rgb_logit = torch.zeros(rgb_feat.shape[0], 3, device=rgb_feat.device) - 100

                # for i, idx_ray in enumerate(ind_uniques):
                #     ray_id_tmp = ray_id == idx_ray
                #     idx_mlp = mlp_idxs[i]
                #     if idx_mlp != 0:
                #         rgb_logit[ray_id_tmp] = self.rgbnets[idx_mlp-1](rgb_feat[ray_id_tmp])
                #print(rgb_logit)
                # time1 = time.time()
                index_mlp = [mlp_idxs[idu].expand(idc) for idu, idc in zip(ind_uniques, ind_counts)]
                index_mlp = torch.cat(index_mlp)
                for part in range(self.num_parts):
                    part_ind = index_mlp == (part + 1)
                    if part_ind.sum() > 0:
                        rgb_logit[part_ind] = self.rgbnets[part](rgb_feat[part_ind])
                # time2 = time.time()
                # print(time2 - time1)
                if self.cfg.dvgo.rgbnet_direct:
                    rgb = torch.sigmoid(rgb_logit)
                else:
                    rgb = torch.sigmoid(rgb_logit + k0_diffuse)
        else:
            rgb_logit = torch.zeros(k0.shape[0], 3, device=k0.device) - 100
            rgb = torch.sigmoid(rgb_logit)
        # time5 = time.time()
        # print(time5 - time4)
        # Ray marching
        rgb_marched = segment_coo(
            src=(weights.unsqueeze(-1) * rgb),
            index=ray_id,
            out=torch.zeros([N, 3], device=weights.device),
            reduce='sum')
        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        # time6 = time.time()
        # print(time6 - time5)

        # time7 = time.time()
        # print(time7 - time6)
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
            # 'index_pred_ray': index_pred_point, # ray_len_unique, num_parts+1
            'index': index_pred,
            # 'ind_uniques': ind_uniques,
            'part_marched': part_marched, # ray_len, num_parts+1
        })
        if render_kwargs.get('render_mask', False):
            mask_marched = segment_coo(
                src=(weights),
                index=ray_id,
                out=torch.zeros([N], device=weights.device),
                reduce='sum')
            ret_dict.update({'mask_marched': mask_marched})
        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                    src=(weights * step_id),
                    index=ray_id,
                    out=torch.zeros([N], device=weights.device),
                    reduce='sum')
            ret_dict.update({'depth': depth})

        return ret_dict

    def export_mesh(self, field, part_fea,
                    # full_reconstruction_dir,
                    # part_reconstruction_dir,
                    resolution=64, mise_resolution=32,
                    padding=1, threshold=0.5, upsamling_steps=3,
                    # name='original',
                    ):
        density = field[0].unsqueeze(0)
        fea = field[1:]
        mesh_generator = MeshGenerator(
            resolution=resolution,
            mise_resolution=mise_resolution,
            padding=padding,
            threshold=threshold,
            upsampling_steps=upsamling_steps,
        )
        model_occupancy_callable = partial(
                self.forward_grid
            )
        model_index_callable = partial(self.index_mlp)
        mesh, part_meshes_list = reconstruct_meshes_from_model(
            model_occupancy_callable,
            model_index_callable,
            mesh_generator,
            chunk_size=50000,
            density=density,
            fea=fea,
            part_fea=part_fea,
            with_parts=True,
            num_parts=self.num_parts,
        )
        return mesh, part_meshes_list
        # try:
        #     mesh, part_meshes_list = reconstruct_meshes_from_model(
        #         model_occupancy_callable,
        #         mesh_generator,
        #         chunk_size,
        #         device,
        #         with_parts=with_parts,
        #         num_parts=config.model.shape_decomposition_network.num_parts,
        #     )
        # except Exception as e:
        #     print("Mesh reconstruction error, skipping...")
        #     continue

        # Export meshes
        # export_meshes_to_path(
        #     full_reconstruction_dir,
        #     part_reconstruction_dir,
        #     mesh,
        #     part_meshes_list,
        #     name
        # )

class SelfAttLayer(nn.Module):
    def __init__(self, dim, heads=4, reduce=1, ffn_dim_mul=2, drop=0.):
        super().__init__()
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.heads = heads
        self.reduce = reduce
        # hidden_dim = dim_head * heads
        ffn_dim = ffn_dim_mul * dim
        self.softmax = nn.Softmax(dim=-1)
        self.q_lin = nn.Linear(dim, dim//reduce)
        self.k_lin = nn.Linear(dim, dim//reduce)
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

    def forward(self, x):
        b, l, c = x.shape
        q = self.q_lin(x)
        k = self.k_lin(x)
        v = self.v_lin(x)
        q = q.view(b, l, self.heads, self.dim_head//self.reduce).transpose(1, 2)  # b, head, l, dim_head
        k = k.view(b, l, self.heads, self.dim_head//self.reduce).transpose(1, 2)
        v = v.view(b, l, self.heads, self.dim_head).transpose(1, 2)

        q = q * self.scale
        k_t = k.transpose(2, 3)  # transpose
        att = self.softmax(q @ k_t)
        v = att @ v  # b, head, l ,dim_head
        v = v.transpose(1, 2).contiguous().view(b, l, c)
        x = x + self.concat_lin(v)
        # x = x + v
        x = self.norm1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x


class CrossAttLayer(nn.Module):
    def __init__(self, dim, heads=4, reduce=4, ffn_dim_mul=2, drop=0.):
        super().__init__()
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.heads = heads
        self.reduce = reduce
        # hidden_dim = dim_head * heads
        ffn_dim = ffn_dim_mul * dim
        self.softmax = nn.Softmax(dim=-1)
        self.q_lin = nn.Linear(dim, dim//reduce)
        self.k_lin = nn.Linear(dim, dim//reduce)
        self.v_lin = nn.Linear(dim, dim)
        # self.concat_lin = nn.Linear(dim, dim)
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
        q = q.view(b, l, self.heads, self.dim_head//self.reduce).transpose(1, 2)  # b, head, l, dim_head
        k = k.view(b, l2, self.heads, self.dim_head//self.reduce).transpose(1, 2)
        v = v.view(b, l2, self.heads, self.dim_head).transpose(1, 2)

        q = q * self.scale
        k_t = k.transpose(2, 3)  # transpose
        att = self.softmax(q @ k_t)
        v = att @ v  # b, head, l ,dim_head
        v = v.transpose(1, 2).contiguous().view(b, l, c)
        # x = x + self.concat_lin(v)
        x = x + v
        x = self.norm1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x



class IndexMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, part_dim=128, n_layers=3):
        super(IndexMLP, self).__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.in_layer_part = nn.Linear(part_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - 2):
            self.hidden_layers.append(CrossAttLayer(hidden_dim))
        self.act = nn.ReLU()
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, part_fea):
        # x: B, 1, 3;    part_fea: B, num_parts, part_fea_dim
        x = self.in_layer(x)
        part_fea = self.in_layer_part(part_fea)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x, part_fea)
        x = self.out_layer(self.act(x))
        return x # B, 1, num_parts

class RelateMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, part_dim=128, n_layers=3):
        super(RelateMLP, self).__init__()
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

class PartAtt(nn.Module):
    def __init__(self, in_dim, out_dim=128, n_layers=3):
        super(PartAtt, self).__init__()
        self.in_layer = nn.Linear(in_dim, out_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.hidden_layers.append(SelfAttLayer(out_dim, reduce=2))

    def forward(self, x):
        # x : num_parts, part_dim
        x = x.unsqueeze(0)
        x = self.in_layer(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return x[0]