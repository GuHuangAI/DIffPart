'''
compared to original diff_nerf_ddm_const_part.py
1. remove the part embedding as condition
2. remove the shape and texture embedding
3. add image as condition
4. fix testing bug
'''
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import os
from diff_nerf.utils import default, identity, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
import random
from einops import rearrange, reduce
from functools import partial
#from diff_nerf.mesh_utils import export_meshes_to_path
#from random import random, randint, sample, choice
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np
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

class DDPM(nn.Module):
    def __init__(
        self,
        model,
        model_n,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        clip_x_start=True,
        train_sample=-1,
        input_keys=['input'],
        start_dist='normal',
        use_l1=False,
        **kwargs
    ):
        super().__init__()
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        only_model = kwargs.pop("only_model", False)
        cfg = kwargs.pop("cfg", None)
        self.model = model
        self.nerf = model_n
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.input_keys = input_keys
        self.cfg = cfg
        self.std_scale = cfg.get('std_scale', 1.)
        self.register_buffer('device_buffer', torch.FloatTensor([cfg.std_scale]))
        self.eps = cfg.get('eps', 1e-4) if cfg is not None else 1e-4
        self.weighting_loss = cfg.get("weighting_loss", False) if cfg is not None else False
        self.use_render_loss = cfg.get('use_render_loss', False)
        self.use_l1 = cfg.get('use_l1', False)
        self.clip_x_start = clip_x_start
        self.image_size = image_size
        self.train_sample = train_sample
        self.objective = objective
        self.start_dist = start_dist
        assert start_dist in ['normal', 'uniform']

        assert objective in {'pred_noise', 'pred_x0', 'pred_v', 'pred_delta', 'pred_KC'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        self.num_timesteps = int(timesteps)
        self.time_range = list(range(self.num_timesteps + 1))
        self.loss_type = loss_type
        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        # helper function to register buffer from float64 to float32
        self.use_l1 = use_l1
        ### define part ###
        self.num_parts = cfg.get("num_parts", 5)
        self.part_fea_dim = cfg.get("part_fea_dim", 64)
        # self.shape_embs = nn.Embedding(cfg.get("num_shape", 500), self.part_fea_dim)
        # self.texture_embs = nn.Embedding(cfg.get("num_shape", 500), self.part_fea_dim)
        # self.part_embeddings = nn.Embedding(self.num_parts, self.part_fea_dim)
        # nn.init.normal_(self.part_embeddings.weight.data, 0.0, 1.0 / math.sqrt(self.part_fea_dim))
        # nn.init.normal_(self.shape_embs.weight.data, 0.0, 1.0 / math.sqrt(self.part_fea_dim))
        # nn.init.normal_(self.texture_embs.weight.data, 0.0, 1.0 / math.sqrt(self.part_fea_dim))
        # self.part_mlp = nn.Linear(self.part_fea_dim, self.part_fea_dim)
        # self.part_shape_mlp = DecNet(self.num_parts, self.part_fea_dim, n_layers=3)
        # self.part_texture_mlp = DecNet(self.num_parts, self.part_fea_dim, n_layers=3)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model)

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if 'ema' in list(sd.keys()):
            sd = sd['ema']
            new_sd = {}
            for k in sd.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]    # remove ema_model.
                    new_sd[new_k] = sd[k]
            sd = new_sd
        else:
            if "model" in list(sd.keys()):
                sd = sd["model"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def get_input(self, batch, return_first_stage_outputs=False, return_original_cond=False):
        assert 'input' in self.input_keys;
        # if len(self.input_keys) > len(batch.keys()):
        #     x, *_ = batch.values()
        # else:
        #     x = batch.values()
        x = batch['input']
        x = x / self.std_scale
        batch['input'] = x
        return batch

    def training_step(self, batch, **kwargs):
        batch = self.get_input(batch)
        loss, loss_dict = self(batch, **kwargs)
        return loss, loss_dict

    def forward(self, batch, *args, **kwargs):
        # continuous time, t in [0, 1]
        x = batch['input']
        eps = self.eps  # smallest time step
        t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
        # t = 0.1 * t
        return self.p_losses(batch, t, *args, **kwargs)

    def q_sample(self, x_start, noise, t, C):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x_noisy = x_start + C * time + torch.sqrt(time) * noise
        return x_noisy

    def pred_x0_from_xt(self, xt, noise, C, t):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x0 = xt - C * time - torch.sqrt(time) * noise
        return x0

    def pred_xt_from_x0(self, x0, noise, C, t):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        xt = x0 + C * time + torch.sqrt(time) * noise
        return xt

    def pred_xtms_from_xt(self, xt, noise, C, t, s):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        s = s.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        mean = xt + C * (time-s) - C * time - s / torch.sqrt(time) * noise
        epsilon = torch.randn_like(mean, device=xt.device)
        sigma = torch.sqrt(s * (time-s) / time)
        xtms = mean + sigma * epsilon
        return xtms

    def p_losses(self, batch, t, noise=None, global_step=1e9, **kwargs):
        x_start = batch['input']
        if self.start_dist == 'normal':
            noise = torch.randn_like(x_start)
        elif self.start_dist == 'uniform':
            noise = 2 * torch.rand_like(x_start) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        part_fea = self.part_mlp(self.part_embeddings.weight)
        C = -1 * x_start             # U(t) = Ct, U(1) = - x0
        x_noisy = self.q_sample(x_start=x_start, noise=noise, t=t, C=C)  # (b, c, h, w)
        pred = self.model(x_noisy, t, cond=part_fea, **kwargs)
        C_pred, noise_pred = pred
        x_rec = self.pred_x0_from_xt(x_noisy, noise_pred, C_pred, t)
        loss_dict = {}
        prefix = 'train'
        target1 = C
        target2 = noise
        target3 = x_start

        loss_simple = 0.
        loss_vlb = 0.
        if self.weighting_loss:
            simple_weight1 = 1 / t.sqrt()
            simple_weight2 = 1 / (1 - t + self.eps).sqrt()
        else:
            simple_weight1 = 1
            simple_weight2 = 1
        loss_simple += simple_weight1 * self.get_loss(C_pred, target1, mean=False).mean([1, 2, 3, 4]) + \
                       simple_weight2 * self.get_loss(noise_pred, target2, mean=False).mean([1, 2, 3, 4])
        if self.use_l1:
            loss_simple += simple_weight1 * (C_pred - target1).abs().mean([1, 2, 3, 4]) + \
                           simple_weight2 * (noise_pred - target2).abs().mean([1, 2, 3, 4])
            loss_simple = loss_simple / 2
        loss_simple = loss_simple.mean()
        loss_dict.update({f'{prefix}/loss_simple': loss_simple})
        rec_weight = (1 - t.reshape(C.shape[0], 1)) ** 2
        render_weight = -torch.log(t.reshape(C.shape[0], 1))
        loss_vlb += torch.abs(x_rec - target3).mean([1, 2, 3, 4]) * rec_weight
        loss_vlb = loss_vlb.mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss = loss_simple + loss_vlb
        if self.cfg.nerf.weight_tv > 0:
            # grid = field[idx].unsqueeze(0)
            # wx = wy = wz = self.cfg.render_kwargs.weight_tv / len(rays_o)
            # total_variation_cuda.total_variation_add_grad(
            #     grid, grid.grad, wx, wy, wz, False)
            grid = x_rec * self.std_scale
            loss_tv = self.total_variation(grid) * self.cfg.nerf.weight_tv
            loss += loss_tv
            loss_dict['loss_tv'] = loss_tv.detach().item()


        if self.use_render_loss and global_step >= self.cfg.get('render_start', 0):
            if global_step >= self.cfg.get("joint_step", 0): # joint optimizing
                render_kwargs = batch["render_kwargs"]
                kwargs['part_fea'] = part_fea  # num_parts, part_fea_dim
                loss_render, loss_render_dict = self.nerf(x_rec * self.std_scale, render_kwargs, loss_weight=render_weight,
                                                          global_step=global_step, joint_learn=True, **kwargs)
                loss_dict.update(loss_render_dict)
                loss += loss_render
            else: # independent optimizing
                render_kwargs = batch["render_kwargs"]
                kwargs['part_fea'] = part_fea.detach()  # num_parts, part_fea_dim
                loss_render, loss_render_dict = self.nerf(x_rec.detach() * self.std_scale, render_kwargs, loss_weight=render_weight,
                                                          global_step=global_step, **kwargs)
                loss_dict.update(loss_render_dict)
                # loss += loss_render
        loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

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

    @torch.no_grad()
    def sample(self, batch_size=16, up_scale=1, cond=None, mask=None, denoise=True):
        image_size, channels = self.image_size, self.channels
        if cond is not None:
            batch_size = cond.shape[0]
        return self.sample_fn((batch_size, channels, image_size[0], image_size[1], image_size[2]),
                              up_scale=up_scale, unnormalize=True, cond=cond, mask=mask, denoise=denoise)

    @torch.no_grad()
    def sample_fn(self, shape, up_scale=1, unnormalize=True, cond=None, mask=None, denoise=False):
        batch, device = shape[0], self.device_buffer.device

        # times = torch.linspace(-1, total_timesteps, steps=self.sampling_timesteps + 1).int()
        # times = list(reversed(times.int().tolist()))
        # time_pairs = list(zip(times[:-1], times[1:]))
        # time_steps = torch.tensor([0.25, 0.15, 0.1, 0.1, 0.1, 0.09, 0.075, 0.06, 0.045, 0.03])
        step = 1. / self.sampling_timesteps
        # time_steps = torch.tensor([0.1]).repeat(10)
        time_steps = torch.tensor([step]).repeat(self.sampling_timesteps)
        if denoise:
            eps = self.eps
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - eps]), torch.tensor([eps])), dim=0)

        if self.start_dist == 'normal':
            img = torch.randn(shape, device=device)
        elif self.start_dist == 'uniform':
            img = 2 * torch.rand(shape, device=device) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        img = F.interpolate(img, scale_factor=up_scale, mode='trilinear', align_corners=True)
        # K = -1 * torch.ones_like(img)
        cur_time = torch.ones((batch,), device=device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((batch,), time_step, device=device)
            if i == time_steps.shape[0] - 1:
                s = cur_time
            if cond is not None:
                pred = self.model(img, cur_time, cond)
            else:
                pred = self.model(img, cur_time)
            # C, noise = pred.chunk(2, dim=1)
            C, noise = pred[:2]
            # correct C
            x0 = self.pred_x0_from_xt(img, noise, C, cur_time)
            C = -1 * x0
            img = self.pred_xtms_from_xt(img, noise, C, cur_time, s)
            # img = self.pred_xtms_from_xt2(img, noise, C, cur_time, s)
            cur_time = cur_time - s
        # img.clamp_(-1., 1.)
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def render_img(self, inputs, render_kwargs):
        input = inputs['input']
        cond = inputs['cond'] if 'cond' in inputs else None
        mask = inputs['mask'] if 'mask' in inputs else None
        rotate_flag = render_kwargs.rotate_flag
        if rotate_flag:
            angle, axes = inputs['rotate_params']
        device = self.device_buffer.device
        H, W, focal = render_kwargs.hwf
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])
        try:
            render_poses = inputs['render_kwargs']['poses']
        except:
            render_pose = torch.stack([pose_spherical(angle, -30.0, 3.0) for angle in np.linspace(-180, 180, 5)[:-1]],
                                      0)
            render_poses = [render_pose for _ in range(input.shape[0])]
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
        reconstructions = self.sample(batch_size=input.shape[0], up_scale=1, cond=cond, mask=mask)
        # cls_logits = self.first_stage_model.classifier(reconstructions)
        # cls_ids = torch.max(cls_logits, 1)[1]
        reconstructions = reconstructions * self.std_scale

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
            # fea = fea + self.first_stage_model.residual(fea.unsqueeze(0))[0]
            render_pose = render_poses[idx_obj]

            # render_kwargs['class_id'] = cls_ids[idx_obj].item()
            for i, c2w in enumerate(render_pose):
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
                    {k: v for k, v in
                     self.nerf.render_train(dens, fea, ro, rd, vd, **render_kwargs).items() if k in keys}
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

    @torch.no_grad()
    def render_img_sample(self, batch_size, render_kwargs, export_mesh=False, **kwargs):
        device = self.device_buffer.device
        H, W, focal = render_kwargs.hwf
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

        render_pose = torch.stack([pose_spherical(angle, -30.0, 3.0) for angle in np.linspace(-180, 180, 10)[:-1]],
                                  0)
        render_poses = [render_pose for _ in range(batch_size)]
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
        reconstructions = self.sample(batch_size=batch_size, up_scale=1, cond=None, mask=None)
        # try:
        #     cls_logits = self.first_stage_model.classifier(reconstructions)
        #     cls_ids = torch.max(cls_logits, 1)[1]
        # except:
        #     cls_ids = None
        reconstructions = reconstructions * self.std_scale

        # reconstructions = input
        rgbss = []
        depthss = []
        bgmapss = []
        for idx_obj in range(reconstructions.shape[0]):

            rgbs = []
            depths = []
            bgmaps = []
            dens = reconstructions[idx_obj][0]
            if export_mesh:
                mesh, part_mesh_list = self.nerf.export_mesh(dens)
            fea = reconstructions[idx_obj][1:]
            # fea = fea + self.first_stage_model.residual(fea.unsqueeze(0))[0]
            render_pose = render_poses[idx_obj]
            # if cls_ids is not None:
            #     render_kwargs['class_id'] = cls_ids[idx_obj].item()
            for i, c2w in enumerate(render_pose):
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
                    {k: v for k, v in
                     self.nerf.render_train(dens, fea, ro, rd, vd, **render_kwargs).items() if k in keys}
                    for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
                ]
                render_result = {
                    k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H, W, -1)
                    for k in render_result_chunks[0].keys()
                }

                if render_kwargs.render_depth:
                    depth = render_result['depth'].permute(2, 0, 1)
                    depths.append(depth)
                rgb = render_result['rgb_marched'].permute(2, 0, 1)
                bgmap = render_result['alphainv_last'].permute(2, 0, 1)
                rgbs.append(rgb)
                bgmaps.append(bgmap)
            rgbss.append(rgbs)
            depthss.append(depths)
            bgmapss.append(bgmaps)
        # rgbs = np.array(rgbs)
        # depths = np.array(depths)
        # bgmaps = np.array(bgmaps)
        # del model
        torch.cuda.empty_cache()
        return rgbss, depthss, bgmapss


class LatentDiffusion(DDPM):
    def __init__(self,
                 auto_encoder,
                 scale_factor=1.0,
                 scale_by_std=True,
                 scale_by_softsign=False,
                 default_scale=False,
                 input_keys=['input', 'cond'],
                 num_timesteps_cond=1,
                 *args,
                 **kwargs
                 ):
        self.scale_by_std = scale_by_std
        self.default_scale = default_scale
        self.scale_by_softsign = scale_by_softsign
        self.num_timesteps_cond = num_timesteps_cond
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        only_model = kwargs.pop("only_model", False)
        super().__init__(*args, **kwargs)
        assert self.num_timesteps_cond <= self.num_timesteps
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        if self.scale_by_softsign:
            self.scale_by_std = False
            print('### USING SOFTSIGN RESCALING')
        assert (self.scale_by_std and self.scale_by_softsign) is False;

        self.init_first_stage(auto_encoder)
        # self.instantiate_cond_stage(cond_stage_config)
        self.input_keys = input_keys
        self.clip_denoised = False

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model)

    def init_first_stage(self, first_stage_model):
        self.first_stage_model = first_stage_model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        # for n, param in self.first_stage_model.named_parameters():
        #     if 'residual' in n or 'rgbnet' in n:
        #         param.requires_grad = True

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def get_first_stage_encoding(self, encoder_posterior):
        z = encoder_posterior.sample()
        return z.detach()

    @torch.no_grad()
    def on_train_batch_start(self, batch):
        # only for the first batch
        if self.scale_by_std and (not self.scale_by_softsign):
            if not self.default_scale:
                assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
                # set rescale weight to 1./std of encodings
                print("### USING STD-RESCALING ###")
                # x, *_ = batch.values()
                encoder_posterior = self.first_stage_model.encode(batch['input'] / self.std_scale)
                z = self.get_first_stage_encoding(encoder_posterior)
                del self.scale_factor
                self.register_buffer('scale_factor', 1. / z.flatten().std())
                print(f"setting self.scale_factor to {self.scale_factor}")
                # print("### USING STD-RESCALING ###")
            else:
                print(f'### USING DEFAULT SCALE {self.scale_factor}')
        else:
            print(f'### USING SOFTSIGN SCALE !')

    @torch.no_grad()
    def get_input(self, batch, return_first_stage_outputs=False, return_original_cond=False):
        assert 'input' in self.input_keys;
        x = batch['input']
        # cond = batch['cond'] if 'cond' in batch else None
        # if cond:
        #     cond = cond / self.first_stage_model.std_scale
        z = self.first_stage_model.encode(x / self.first_stage_model.std_scale)
        z = self.get_first_stage_encoding(z)
        #print(1. / z.flatten().std())
        #exit()
        batch['input'] = z
        # out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.first_stage_model.decode(z)
            # out.extend([x, xrec])
            batch['rec'] = xrec
        if return_original_cond:
            batch['ori'] = x
        return batch

    def training_step(self, batch, *args, **kwargs):
        batch = self.get_input(batch)
        # print(batch['input'].std())
        if self.scale_by_softsign:
            batch['input'] = F.softsign(batch['input'])
        elif self.scale_by_std:
            batch['input'] = self.scale_factor * batch['input']
        # print('grad', self.scale_bias.grad)
        loss, loss_dict = self(batch, *args, **kwargs)
        return loss, loss_dict

    def forward(self, batch, *args, **kwargs):
        # continuous time, t in [0, 1]
        eps = self.eps  # smallest time step
        x = batch['input']
        t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
        # kwargs = batch
        return self.p_losses(batch, t, *args, **kwargs)

    def p_losses(self, batch, t, noise=None, global_step=1e9, **kwargs):
        x_start = batch['input']
        idx = batch['obj_idx']
        img_cond = batch['ref_img']
        if self.start_dist == 'normal':
            noise = torch.randn_like(x_start)
        elif self.start_dist == 'uniform':
            noise = 2 * torch.rand_like(x_start) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        # K = -1. * torch.ones_like(x_start)
        # C = noise - x_start  # t = 1000 / 1000
        # part_shape_fea = self.part_shape_mlp(self.shape_embs.weight[idx]) # B, num_parts, part_fea_dim
        # part_texture_fea = self.part_texture_mlp(self.texture_embs.weight[idx]) # B, num_parts, part_fea_dim
        C = -1 * x_start  # U(t) = Ct, U(1) = - x0
        x_noisy = self.q_sample(x_start=x_start, noise=noise, t=t, C=C)  # (b, 2, c, h, w)
        pred = self.model(x_noisy, t, cond1=None, cond2=None, image=img_cond, **kwargs)
        C_pred, noise_pred = pred
        # C_pred = C_pred / torch.sqrt(t)
        # noise_pred = noise_pred / torch.sqrt(1 - t)
        x_rec = self.pred_x0_from_xt(x_noisy, noise_pred, C_pred, t)
        loss_dict = {}
        prefix = 'train'

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_delta':
            if t.shape[-1] == self.num_timesteps:
                target = x_start - x_noisy[:, -1]
            else:
                target = self.q_sample(x_start=x_start, t=t[:, -1] - 1, noise=noise) - x_noisy[:, -1]
        elif self.objective == 'pred_KC':
            target1 = C
            target2 = noise
            target3 = x_start
        elif self.objective == 'pred_v':
            target = C
        else:
            raise NotImplementedError()
        loss_simple = 0.
        loss_vlb = 0.
        if self.weighting_loss:
            # simple_weight1 = torch.exp(t)
            # simple_weight2 = torch.exp(torch.sqrt(1 - t))
            simple_weight1 = 1 / t #.sqrt()
            simple_weight2 = 1 / (1 - t + self.eps)#.sqrt()
        else:
            simple_weight1 = 1
            simple_weight2 = 1
        loss_simple += simple_weight1 * self.get_loss(C_pred, target1, mean=False).mean([1, 2, 3, 4]) + \
                       simple_weight2 * self.get_loss(noise_pred, target2, mean=False).mean([1, 2, 3, 4])
        if self.use_l1:
            loss_simple += simple_weight1 * (C_pred - target1).abs().mean([1, 2, 3, 4]) + \
                       simple_weight2 * (noise_pred - target2).abs().mean([1, 2, 3, 4])
            loss_simple = loss_simple / 2
        loss_simple = loss_simple.mean()
        loss_dict.update({f'{prefix}/loss_simple': loss_simple})
        rec_weight = (1 - t.reshape(C.shape[0], 1)) ** 2
        render_weight = -torch.log(t.reshape(C.shape[0], 1))
        loss_vlb += torch.abs(x_rec - target3).mean([1, 2, 3, 4]) * rec_weight
        # if self.perceptual_weight > 0.:
        #     loss_vlb += self.perceptual_loss(x_rec, target3).mean([1, 2, 3])
        loss_vlb = loss_vlb.mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss = loss_simple + loss_vlb

        # loss += 0.0001 * torch.norm(self.shape_embs.weight, dim=-1).mean()
        # loss += 0.0001 * torch.norm(self.texture_embs.weight, dim=-1).mean()

        with torch.no_grad():
            x_rec_dec = self.first_stage_model.decode(x_rec / self.scale_factor)
        # class_id = batch["class_id"]
        SpecifyGradient2.apply(x_rec, x_rec_dec)
        if self.cfg.nerf.weight_tv > 0:
            grid = x_rec_dec * self.std_scale
            loss_tv = self.total_variation(grid) * self.cfg.nerf.weight_tv
            loss += loss_tv
            loss_dict['loss_tv'] = loss_tv.detach().item()
        # if self.cfg.first_stage.use_cls_loss and global_step >= self.cfg.first_stage.get('cls_start', 0):
        #     grad_cls, cls_loss_item, acc = self.class_loss(x_rec_dec, class_id, loss_weight=rec_weight)
        #     cls_loss = SpecifyGradient.apply(x_rec, grad_cls)
        #     loss += cls_loss.mean()
        #     loss_dict.update({'cls_loss': cls_loss_item, 'acc': acc})
        if self.use_render_loss and global_step >= self.cfg.get('render_start', 0):
            if global_step >= self.cfg.get("joint_step", 0): # joint optimizing
                render_kwargs = batch["render_kwargs"]
                # kwargs['part_shape_fea'] = part_shape_fea  # B, num_parts, part_fea_dim
                # kwargs['part_texture_fea'] = part_texture_fea  # B, num_parts, part_fea_dim
                loss_render, loss_render_dict = self.nerf(x_rec_dec * self.std_scale, render_kwargs, loss_weight=render_weight,
                                                          global_step=global_step, joint_learn=True, **kwargs)
                loss_dict.update(loss_render_dict)
                loss += loss_render
            else: # independent optimizing
                render_kwargs = batch["render_kwargs"]
                # kwargs['part_shape_fea'] = part_shape_fea.detach()  # B, num_parts, part_fea_dim
                # kwargs['part_texture_fea'] = part_texture_fea.detach()  # B, num_parts, part_fea_dim
                loss_render, loss_render_dict = self.nerf(x_rec_dec.detach() * self.std_scale, render_kwargs, loss_weight=render_weight,
                                                          global_step=global_step, **kwargs)
                loss_dict.update(loss_render_dict)

        return loss, loss_dict

    def render_loss(self, field, render_kwargs, **kwargs):
        HWs, Kss, nears, fars, i_trains, i_vals, i_tests, posess, imagess = [
            render_kwargs[k] for k in [
                'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'images'
            ]
        ]
        loss = 0.
        count = 0
        psnr = 0.
        densities = field[:, 0]
        features = field[:, 1:]
        class_ids = kwargs['class_id']
        loss_weight = kwargs['loss_weight'] if 'loss_weight' in kwargs else torch.ones_like(class_ids)
        for idx, (dens, fea, HW, Ks, near, far, i_train, i_val, i_test, poses, images, cls_id, lw) in \
                enumerate(zip(densities, features, HWs, Kss, nears, fars, i_trains, i_vals, i_tests, posess, imagess,
                              class_ids, loss_weight)):
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
                'bg': self.cfg.nerf.bg,
                'rand_bkgd': False,
                'stepsize': self.cfg.nerf.stepsize,
                'inverse_y': self.cfg.nerf.inverse_y,
                'flip_x': self.cfg.nerf.flip_x,
                'flip_y': self.cfg.nerf.flip_y,
                'class_id': cls_id
            }

            # rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = self.first_stage_model.get_training_rays_in_maskcache_sampling(
            #     rgb_tr_ori=rgb_tr_ori,
            #     train_poses=poses,
            #     HW=HW, Ks=Ks,
            #     ndc=False, inverse_y=False,
            #     flip_x=False, flip_y=False,
            #     density=dens,
            #     render_kwargs=render_kwarg_train)
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = self.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses,
                HW=HW, Ks=Ks,
                ndc=False, inverse_y=render_kwarg_train['inverse_y'],
                flip_x=render_kwarg_train['flip_x'], flip_y=render_kwarg_train['flip_y'],
            )
            # render_kwarg.update()
            for iter in range(self.cfg.nerf.inner_iter):
                sel_b = torch.randint(rgb_tr.shape[0], [self.cfg.nerf.N_rand])
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
                render_result = self.first_stage_model.render_train(dens, fea, rays_o, rays_d, viewdirs, **render_kwarg_train)
                loss_main = self.cfg.first_stage.render_kwargs.weight_main * F.mse_loss(render_result['rgb_marched'], target)
                psnr_cur = -10. * torch.log10(loss_main.detach() / self.cfg.first_stage.render_kwargs.weight_main)
                psnr += psnr_cur
                pout = render_result['alphainv_last'].clamp(1e-6, 1 - 1e-6)
                entropy_last_loss = -(pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)).mean()
                loss_entropy_last = self.cfg.first_stage.render_kwargs.weight_entropy_last * entropy_last_loss
                rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
                rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
                loss_rgbper = self.cfg.first_stage.render_kwargs.weight_rgbper * rgbper_loss
                loss += lw.item() * (loss_main + loss_entropy_last + loss_rgbper)
            count += 1
            torch.cuda.empty_cache()
        return loss / count, loss.detach().item() / count, \
            psnr.item() / count / self.cfg.first_stage.render_kwargs.inner_iter

    def class_loss(self, field, classes_id, **kwargs):
        loss_weight = kwargs['loss_weight'] if 'loss_weight' in kwargs else torch.ones_like(classes_id)
        b = field.shape[0]
        logits = self.first_stage_model.classifier(field)   # N， C
        # loss_class = F.cross_entropy(logits, classes_id) * self.cfg.first_stage.render_kwargs.weight_cls
        for (logit, class_id, lw) in zip(logits, classes_id, loss_weight):
            loss_class = F.cross_entropy(logit, class_id) * lw.item()
        # print(loss_class.shape)
        acc = (torch.max(logits, 1)[1].view(classes_id.size()).data == classes_id.data).sum() / b
        return loss_class, loss_class.detach().item(), acc.item() * 100

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss


    @torch.no_grad()
    def sample(self, batch_size=16, up_scale=1, cond=None, mask=None, denoise=True, device=None,
               part_shape_fea=None, part_texture_fea=None, image=None):
        image_size, channels = self.image_size, self.channels
        if cond is not None:
            batch_size = cond.shape[0]
        down_ratio = self.first_stage_model.down_ratio
        z = self.sample_fn((batch_size, channels, image_size[0] // down_ratio, image_size[1] // down_ratio, image_size[2] // down_ratio),
                           up_scale=up_scale, unnormalize=False, cond=cond, denoise=denoise, device=device,
                           part_shape_fea=part_shape_fea, part_texture_fea=part_texture_fea, image=image)

        if self.scale_by_std:
            z = 1. / self.scale_factor * z.detach()
        elif self.scale_by_softsign:
            z = z / (1 - z.abs())
            z = z.detach()
        # print(z.shape)
        x_rec = self.first_stage_model.decode(z)
        # x_rec = unnormalize_to_zero_to_one(x_rec)
        # x_rec = torch.clamp(x_rec, min=0., max=1.)
        if mask is not None:
            x_rec = mask * cond + (1 - mask) * x_rec
        return x_rec

    @torch.no_grad()
    def sample_fn(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False,
                  device=None, part_shape_fea=None, part_texture_fea=None, image=None):
        batch, device, total_timesteps, sampling_timesteps, objective = shape[0], \
            device, self.num_timesteps, self.sampling_timesteps, self.objective

        # times = torch.linspace(-1, total_timesteps, steps=self.sampling_timesteps + 1).int()
        # times = list(reversed(times.int().tolist()))
        # time_pairs = list(zip(times[:-1], times[1:]))
        # time_steps = torch.tensor([0.25, 0.15, 0.1, 0.1, 0.1, 0.09, 0.075, 0.06, 0.045, 0.03])
        step = 1. / self.sampling_timesteps
        # time_steps = torch.tensor([0.1]).repeat(10)
        time_steps = torch.tensor([step]).repeat(self.sampling_timesteps)
        if denoise:
            eps = self.eps
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - eps]), torch.tensor([eps])), dim=0)

        if self.start_dist == 'normal':
            img = torch.randn(shape, device=device)
        elif self.start_dist == 'uniform':
            img = 2 * torch.rand(shape, device=device) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        img = F.interpolate(img, scale_factor=up_scale, mode='trilinear', align_corners=True)
        # K = -1 * torch.ones_like(img)
        cur_time = torch.ones((batch,), device=device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((batch,), time_step, device=device)
            if i == time_steps.shape[0] - 1:
                s = cur_time
            if cond is not None:
                pred = self.model(img, cur_time, cond)
            else:
                pred = self.model(img, cur_time, cond1=part_shape_fea, cond2=part_texture_fea, image=image)
            # C, noise = pred.chunk(2, dim=1)
            C, noise = pred[:2]
            if self.scale_by_softsign:
                # correct the C for softsign
                x0 = self.pred_x0_from_xt(img, noise, C, cur_time)
                x0 = torch.clamp(x0, min=-0.987654321, max=0.987654321)
                C = -x0
            # correct C
            x0 = self.pred_x0_from_xt(img, noise, C, cur_time)
            C = -1 * x0
            img = self.pred_xtms_from_xt(img, noise, C, cur_time, s)
            # img = self.pred_xtms_from_xt2(img, noise, C, cur_time, s)
            cur_time = cur_time - s
        if self.scale_by_softsign:
            img.clamp_(-0.987654321, 0.987654321)
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def render_img_sample(self, batch_size, render_kwargs, export_mesh=False, **kwargs):
        device = self.device_buffer.device
        input = kwargs['input']
        ref_img = input['ref_img']
        H, W, focal = render_kwargs.hwf
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

        render_pose = torch.stack([pose_spherical(angle, -30.0, 3.0) for angle in np.linspace(-180, 180, 10)[:-1]],
                                  0)
        render_poses = [render_pose for _ in range(batch_size)]
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
        # idx = random.sample(list(range(200)), batch_size)
        idx = input['obj_idx']
        # part_shape_fea = self.part_shape_mlp(self.shape_embs.weight[idx])  # B, num_parts, part_fea_dim
        # part_texture_fea = self.part_texture_mlp(self.texture_embs.weight[idx])  # B, num_parts, part_fea_dim
        reconstructions = self.sample(batch_size=batch_size, up_scale=1, cond=None, mask=None, device=device,
                                      part_shape_fea=None, part_texture_fea=None, image=ref_img)
        # try:
        #     cls_logits = self.first_stage_model.classifier(reconstructions)
        #     cls_ids = torch.max(cls_logits, 1)[1]
        # except:
        #     cls_ids = None
        reconstructions = reconstructions * self.std_scale
        # reconstructions = input
        rgbss = []
        partss = []
        depthss = []
        bgmapss = []
        meshes = []
        part_mesh_lists = []
        for idx_obj in range(reconstructions.shape[0]):
            if export_mesh:
                mesh, part_mesh_list = self.nerf.export_mesh(reconstructions[idx_obj], part_fea)
                meshes.append(mesh)
                part_mesh_lists.append(part_mesh_list)
            rgbs = []
            parts = []
            depths = []
            bgmaps = []
            dens = reconstructions[idx_obj][0]
            fea = reconstructions[idx_obj][1:]
            # fea = fea + self.first_stage_model.residual(fea.unsqueeze(0))[0]
            render_pose = render_poses[idx_obj]
            # if cls_ids is not None:
            #     render_kwargs['class_id'] = cls_ids[idx_obj].item()
            for i, c2w in enumerate(render_pose):
                # H, W = HW[i]
                # K = Ks[i]
                # H, W = HW
                # K = Ks
                c2w = torch.Tensor(c2w)
                rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, ndc, inverse_y=render_kwargs.inverse_y,
                    flip_x=render_kwargs.flip_x, flip_y=render_kwargs.flip_y)
                keys = ['rgb_marched', 'depth', 'alphainv_last', 'part_marched']
                rays_o = rays_o.flatten(0, -2).to(device)
                rays_d = rays_d.flatten(0, -2).to(device)
                viewdirs = viewdirs.flatten(0, -2).to(device)
                render_result_chunks = [
                    {k: v for k, v in
                     self.nerf.render_train(dens, fea, ro, rd, vd, **render_kwargs).items() if k in keys}
                    for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
                ]
                render_result = {
                    k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H, W, -1)
                    for k in render_result_chunks[0].keys()
                }

                if render_kwargs.render_depth:
                    depth = render_result['depth'].permute(2, 0, 1)
                    depths.append(depth)
                rgb = render_result['rgb_marched'].permute(2, 0, 1)
                bgmap = render_result['alphainv_last'].permute(2, 0, 1)
                part = render_result['part_marched'].permute(2, 0, 1)
                part = torch.argmax(part, dim=0, keepdim=True).to(torch.float)
                colors = torch.Tensor([
                    [1, 1, 1],  #
                    [0, 0, 1],  #
                    [1, 0, 0],  #
                    [0, 1, 0],  #
                    [1, 1, 0]
                    # Add more colors for other classes as needed
                ])
                part_rgb = torch.ones((part.shape[1], part.shape[2], 3))
                # part = part_rgb.expand(3, part.shape[0], part.shape[1])
                for i in range(len(colors)):
                    mask = torch.all(part == i, dim=0)
                    # mask = part == i
                    part_rgb[mask] = colors[i]
                parts.append(part_rgb.permute(2, 0, 1))
                rgbs.append(rgb)
                bgmaps.append(bgmap)
            rgbss.append(rgbs)
            partss.append(parts)
            depthss.append(depths)
            bgmapss.append(bgmaps)
        # rgbs = np.array(rgbs)
        # depths = np.array(depths)
        # bgmaps = np.array(bgmaps)
        # del model
        torch.cuda.empty_cache()
        if export_mesh:
            return rgbss, depthss, bgmapss, partss, meshes, part_mesh_lists
        return rgbss, depthss, bgmapss, partss, meshes, part_mesh_lists

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones(input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype)
        # return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

class SpecifyGradient2(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, out_tensor):
        ctx.save_for_backward(out_tensor)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones(input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype)
        # return out_tensor

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        # pdb.set_trace()
        gt_grad = grad_scale
        return gt_grad, None

class PartAtt(nn.Module):
    def __init__(self, in_dim, out_dim=128, n_layers=3):
        super(PartAtt, self).__init__()
        # self.in_layer = nn.Linear(in_dim, out_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.hidden_layers.append(SelfAttLayer(in_dim, reduce=1))

    def forward(self, x):
        # x : b, num_parts, part_dim
        # x = self.in_layer(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return x

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

class DecNet(nn.Module):
    def __init__(self, num_parts, part_fea_dim, n_layers=4):
        super(DecNet, self).__init__()
        self.num_parts = num_parts
        self.part_fea_dim = part_fea_dim
        self.mlp = nn.Linear(part_fea_dim, num_parts * part_fea_dim)
        self.att = PartAtt(self.part_fea_dim, self.part_fea_dim, n_layers=n_layers)

    def forward(self, x):
        x = self.mlp(x).reshape(-1, self.num_parts, self.part_fea_dim)
        x = self.att(x)
        return x