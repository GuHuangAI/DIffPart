import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import torch.distributions.categorical as cate
from torch_scatter import segment_coo
from diff_nerf import dvgo, grid
from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

####   DVGO NeRF  ####
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
        self.num_parts = self.cfg.get('num_parts', 8)
        self.part_fea_dim = self.cfg.get('part_fea_dim', 128)
        self.part_embeddings = nn.Embedding(self.num_parts, self.part_fea_dim)
        nn.init.normal_(self.part_embeddings.weight.data, 0.0, 1.0 / math.sqrt(self.part_fea_dim))
        self.part_mlp = nn.Linear(self.part_fea_dim, self.part_fea_dim)
        # dim_index = 3 + 3 * viewbase_pe * 2
        self.index_mlp = IndexMLP(in_dim=self.cfg.dvgo.rgbnet_dim, out_dim=self.num_parts, part_dim=self.part_fea_dim,
                                  hidden_dim=self.part_fea_dim)

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
                'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'images', 'masks'
            ]
        ]
        loss = 0.
        # loss_item = 0.
        psnr = 0.
        bs = field.shape[0]
        densities = field[:, 0]
        features = field[:, 1:]
        accelerator = kwargs['accelerator']
        # features = features + self.residual(features)
        loss_weights = kwargs['loss_weight'] if 'loss_weight' in kwargs else torch.ones(len(field),)
        for dens, fea, HW, Ks, near, far, i_train, i_val, i_test, poses, images, masks, lw in \
                zip(densities, features, HWs, Kss, nears, fars, i_trains, i_vals, i_tests, posess, imagess, maskss, loss_weights):
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
            for iter in range(self.cfg.inner_iter):
                # loss = 0.
                sel_b = torch.randint(rgb_tr.shape[0], [self.cfg.N_rand])
                target = rgb_tr[sel_b].to(device)
                target_m = mask_tr[sel_b].to(device)
                rays_o = rays_o_tr[sel_b].to(device)
                rays_d = rays_d_tr[sel_b].to(device)
                viewdirs = viewdirs_tr[sel_b]
                render_result = self.render_train(dens, fea, rays_o, rays_d, viewdirs, **render_kwarg_train)
                loss_main = self.cfg.weight_main * F.mse_loss(render_result['rgb_marched'], target)
                psnr_cur = -10. * torch.log10(loss_main.detach() / self.cfg.weight_main)
                psnr += psnr_cur
                pout = render_result['alphainv_last'].clamp(1e-6, 1 - 1e-6)
                entropy_last_loss = -(pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)).mean()
                loss_entropy_last = self.cfg.weight_entropy_last * entropy_last_loss
                rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
                rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
                loss_rgbper = self.cfg.weight_rgbper * rgbper_loss
                loss_ind_entropy = self.cfg.weight_ind_en * self.loss_entropy(render_result['index'])
                # loss += loss_main + loss_entropy_last + loss_rgbper
                # torch.cuda.empty_cache()
                loss += loss_main + loss_entropy_last + loss_rgbper + loss_ind_entropy
                if mask_tr is not None:
                    loss_mask = self.cfg.weight_mask * F.mse_loss(render_result['mask_marched'], target_m)
                    loss_comparable = self.cfg.weight_comparable * self.loss_comparable(
                                                    render_result['index'], render_result['ray_id'], target_m)
                    # loss_coverage = self.cfg.weight_coverage * self.loss_coverage(render_result['index_value'], target_m)
                    loss += loss_mask + loss_comparable # + loss_coverage
                loss = loss * lw.item()
                # loss_item += loss.detach().item()
                # accelerator.backward(loss)
        # loss_item = loss.detach().item()
        loss_dict = {
            'loss_render_main': loss_main.detach().item(),
            'psnr': psnr/bs/self.cfg.inner_iter,
            'loss_ind_entropy': loss_ind_entropy.detach().item(),
        }
        if mask_tr is not None:
            loss_dict['loss_mask'] = loss_mask.detach().item()
            loss_dict['loss_comparable'] = loss_comparable.detach().item()
        #return loss/bs/self.cfg.inner_iter, loss_item/bs/self.cfg.inner_iter, psnr/bs/self.cfg.inner_iter
        return loss / bs / self.cfg.inner_iter, loss_dict

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

        return

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
            mask_tr = torch.zeros([N, ], device=DEVICE)
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
                mask_tr[top:top + n].copy_(mas[mask])
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
            mask_tr = torch.zeros([N, ], device=DEVICE)
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
                mask_tr[top:top + n].copy_(mas.flatten(0, 1))
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
            # density = density[mask]
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
        batch = ray_pts.shape[0]
        k0 = self.forward_grid(fea, ray_pts)
        # rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        # xyz_emb = (rays_xyz.unsqueeze(-1) * self.viewfreq).flatten(-2)
        # xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)

        part_fea = self.part_mlp(self.part_embeddings.weight)
        # index_pred = self.index_mlp(xyz_emb.unsqueeze(1), part_fea[None, ::].repeat(batch, 1, 1)).squeeze(1)
        index_pred = [self.index_mlp(in1, in2) for in1, in2 in zip(k0.unsqueeze(1).split(8192, 0), \
                                                       part_fea[None, ::].repeat(batch, 1, 1).split(8192, 0)
                                                    )]
        index_pred = torch.cat(index_pred, dim=0).squeeze(1)
        # index_value = index_pred[:, -1]
        # index_pred = index_pred[:, :-1]
        # ind_uniques = torch.unique(ray_id)
        # index_value_pred = torch.zeros(len(ind_uniques), device=index_pred.device) # N,
        # for i, idx in enumerate(ind_uniques):
        #     id_temp = ray_id == idx
        #     index_value_temp = index_value[id_temp]
        #     index_value_pred[i] = index_value_temp.max()
        index_mlp = torch.max(index_pred, dim=-1)[1]

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
                                                       part_fea[None, ::].repeat(batch, 1, 1).split(8192, 0)
                                                    )]
            rgb_feat = torch.cat(rgb_feat, dim=0).squeeze(1)
            rgb_logit = torch.zeros(rgb_feat.shape[0], 3, device=rgb_feat.device) - 100
            for part in range(self.num_parts):
                part_ind = index_mlp == part
                if part_ind.sum() > 0 :
                    rgb_logit[part_ind] = self.rgbnets[part](rgb_feat[part_ind])
            if self.cfg.dvgo.rgbnet_direct:
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
            # 'index_value': index_value_pred,
            'index': index_pred,
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
        x = self.in_layer(x)
        part_fea = self.in_layer_part(part_fea)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x, part_fea)
        x = self.out_layer(self.act(x))
        return x

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

