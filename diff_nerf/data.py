import torch
import os
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
from pathlib import Path
import json
from functools import partial
from bisect import bisect_left, bisect_right
import collections
from torch._six import string_classes
import re
import open3d
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
import glob
import random
import math
import cv2
import imageio
np_str_obj_array_pattern = re.compile(r'[SaUO]')
# angle_list = [0, 45, 90, 135, 180, 225, 270, 315, ] # not 90 * n may generate error reconstruction
angle_list = [0, 90, 180, 270, ]  # rotate angles
axes_list = [0, 1, 2]  # rotate axes
# axes_list = [(1, 0), (2, 0), (2, 1)]
cls_maps = {
    '02747177': 0, '02801938': 1, '02818832': 2,
    '02876657': 3, '03001627': 4, '03337140': 5,
    '03636649': 6, '03991062': 7, '04256520': 8,
    '04379243': 9,
}

def draw_density(density, volume_size=64):
    cache_grid_xyz = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, volume_size),
        torch.linspace(-1, 1, volume_size),
        torch.linspace(-1, 1, volume_size),
    ), -1)
    # cache_grid_xyz = torch.stack(torch.meshgrid(
    #     torch.linspace(0, 64, volume_size),
    #     torch.linspace(0, 64, volume_size),
    #     torch.linspace(0, 64, volume_size),
    # ), -1)
    g_sample = F.grid_sample(density.unsqueeze(0), cache_grid_xyz.unsqueeze(0),
                             mode='bilinear', align_corners=True).reshape(-1)
    # cache_grid_xyz = cache_grid_xyz.reshape(-1, 3).long()
    cache_grid_xyz = cache_grid_xyz.reshape(-1, 3)
    # g_sample = density[0, cache_grid_xyz[:, 0], cache_grid_xyz[:, 1], cache_grid_xyz[:, 2]]
    cache_grid_norm = torch.tensor([0, 0, 1]).repeat(cache_grid_xyz.shape[0], 1)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(cache_grid_xyz)
    # pcd.normals = open3d.utility.Vector3dVector(cache_grid_norm)
    g_sample[g_sample < 0] = 0
    max_label = g_sample.max()
    min_label = g_sample.min()
    g_sample = (g_sample - min_label) / (max_label - min_label)
    # g_sample[g_sample < 0.5] = 0
    # g_sample[g_sample > 0.5] = 1
    colors = plt.get_cmap("Blues")(g_sample)
    pcd.colors = open3d.utility.Vector3dVector(colors[:, :3])
    open3d.visualization.draw_geometries([pcd], 'part of cloud', width=500, height=500,
                                         mesh_show_wireframe=True)

def max_min_normalize(x, maxm, minm, new_maxm=1, new_minm=-1):
    x = (x - minm) / (maxm - minm) * (new_maxm - new_minm) + new_minm
    return x

def max_min_unnormalize(x, maxm, minm, new_maxm=1, new_minm=-1):
    x = (x - new_minm) / (new_maxm - new_minm) * (maxm - minm) + minm
    return x

class VolumeDataset(data.Dataset):
    def __init__(self, tar_path, image_path, load_rgb_net=False, maxm=191.3546, minm=-258.9259,
                 use_rotate_transform=True, load_mask_cache=False, scale_factor=1, normalize=False,
                 sample_num=5, split='train', load_render_kwargs=False, load_mask=False,
                 cfg={}, **kwargs):
        super(VolumeDataset, self).__init__()
        self.tar_path = Path(tar_path)
        self.image_path = Path(image_path)
        self.sample_num = sample_num
        self.split = split
        self.maxm = maxm
        self.minm = minm
        # self.cls_names = os.listdir(self.image_path)
        self.obj_names = {}
        self.obj_total_num = 0
        self.obj_nums = []
        self.normalize = normalize
        # self.cls_names = os.listdir(self.tar_path)
        self.cls_names = cfg.get('cls_names', ['03001627'])
        for cls_name in self.cls_names:
            self.obj_names[cls_name] = os.listdir(self.tar_path / cls_name)
            self.obj_total_num += len(os.listdir(self.tar_path / cls_name))
            self.obj_nums.append(self.obj_total_num)

        self.load_rgb_net = load_rgb_net
        self.load_mask_cache = load_mask_cache
        self.scale_factor = scale_factor
        self.cfg = cfg
        self.half_res = cfg.get('half_res', False)
        self.white_bkgd = cfg.get('white_bkgd', False)

        self.use_rotate_transform = use_rotate_transform
        self.load_render_kwargs = load_render_kwargs
        self.load_mask = load_mask

    def __len__(self):
        return self.obj_total_num

    def find_idx(self, idx):
        assert idx < self.obj_nums[-1];
        cls_idx = bisect_left(self.obj_nums, idx+1)
        if cls_idx == 0:
            obj_idx = idx
        else:
            obj_idx = idx - self.obj_nums[cls_idx-1]
        return cls_idx, obj_idx

    def __getitem__(self, idx):
        # print(idx)
        res = {}
        cls_idx, obj_idx = self.find_idx(idx)
        cls_name = self.cls_names[cls_idx]
        obj_name = self.obj_names[cls_name][obj_idx]
        obj_tar_path = self.tar_path / cls_name / obj_name / 'fine_last.tar'
        obj_tar = torch.load(str(obj_tar_path), map_location=lambda storage, loc: storage)
        obj_weight = obj_tar['model_state_dict']
        obj_kwargs = obj_tar['model_kwargs']
        obj_kwargs['cls_name'] = cls_name
        obj_kwargs['obj_name'] = obj_name
        obj_kwargs['maxm'] = self.maxm
        obj_kwargs['minm'] = self.minm
        density = obj_weight['density.grid'].detach().squeeze(0)
        # alpha = 1 - (1 + torch.exp(density - 8.5951)) ** (-0.25)
        # density = nn.functional.softsign(density)
        feat = obj_weight['k0.grid'].detach().squeeze(0)
        field = torch.cat([density, feat], dim=0)

        if self.use_rotate_transform:
            '''
            angle = random.choice(angle_list)
            axes = random.choice(axes_list)
            density = torch.from_numpy(ndimage.rotate(density[0], angle=angle, axes=axes, reshape=False)[None,::])
            for i in range(feat.shape[0]):
                feat[i] = torch.from_numpy(ndimage.rotate(feat[i], angle=angle, axes=axes, reshape=False))
            res['rotate_params'] = (angle, axes)
            '''
            ### use torch rotate to make differential ###
            angle = random.choice(angle_list)
            angle = angle / 180 * math.pi
            axes = random.choice(axes_list)
            if axes == 0:
                transform_matrix = torch.tensor([
                    [1, 0, 0, 0],
                    [0, math.cos(angle), math.sin(-angle), 0],
                    [0, math.sin(angle), math.cos(angle), 0],
                ])
            elif axes == 1:
                transform_matrix = torch.tensor([
                    [math.cos(angle), 0, math.sin(angle), 0],
                    [0, 1, 0, 0],
                    [math.sin(-angle), 0, math.cos(angle), 0],
                ])
            elif axes == 2:
                transform_matrix = torch.tensor([
                    [math.cos(angle), math.sin(-angle), 0, 0],
                    [math.sin(angle), math.cos(angle), 0, 0],
                    [0, 0, 1, 0],
                ])
            else:
                raise ValueError("Do not support the axes {}!".format(axes))
            grid = F.affine_grid(transform_matrix.unsqueeze(0), field.unsqueeze(0).shape)
            field = F.grid_sample(field.unsqueeze(0), grid, mode='nearest')[0]

        if self.normalize:
            field = max_min_normalize(field, self.maxm, self.minm)
        # draw_density(density)
        # field = torch.cat([density, feat], dim=0)
        # field = nn.functional.interpolate(field.unsqueeze(0),
        #                                   scale_factor=self.scale_factor, mode='trilinear').squeeze(0)
        # field = nn.functional.softsign(field)

        res['kwargs'] = obj_kwargs
        res['input'] = field
        try:
            res['class_id'] = cls_maps[cls_name]
        except:
            res['class_id'] = 0
        if self.load_rgb_net:
            rgb_net_weight = {}
            for k in obj_weight.keys():
                if k.startswith('rgbnet'):
                    rgb_net_weight[k] = obj_weight[k]
            res['rgb_net_weight'] = rgb_net_weight
        if self.load_mask_cache:
            mask_cache_data = obj_weight['mask_cache.mask'].float().unsqueeze(0)
            # mask_cache_data = nn.functional.interpolate(
            #     mask_cache_data.unsqueeze(0).unsqueeze(0),
            #     scale_factor=self.scale_factor, mode='nearest').squeeze(0).squeeze(0).to(torch.bool)
            res['mask_cache'] = mask_cache_data
        if self.load_render_kwargs:
            data_type = self.cfg.get('data_type', 'blender')
            if data_type == "blender":
                image_path = self.image_path / cls_name / obj_name / 'images'
            elif data_type == "srn":
                image_path = self.image_path / cls_name / obj_name
            else:
                raise NotImplementedError
            render_kwargs = load_data(image_path, sample_num=self.sample_num, white_bkgd=self.white_bkgd,
                                      split=self.split, half_res=self.half_res, data_type=data_type,
                                      load_mask=self.load_mask)
            res['render_kwargs'] = render_kwargs
        return res

def load_blender_data(basedir, half_res=False, sample_num=5, split='train'):
    assert split in ['train', 'val', 'test'];
    metas = {}
    # for s in splits:
    #     with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
    #         metas[s] = json.load(fp)
    with open(os.path.join(basedir, 'transforms_{}.json'.format(split)), 'r') as fp:
        metas[split] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_depths = []
    counts = [0]
    for s in [split]:
        meta = metas[s]
        imgs = []
        poses = []
        depths = []
        # if s=='train' or testskip==0:
        #     skip = 1
        # else:
        #     skip = testskip
        # for frame in meta['frames'][::skip]:
        if sample_num > 0:
            frames = random.sample(meta['frames'], sample_num)
        for frame in frames:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            fname_d = os.path.join(basedir, frame['file_path'][:-6] + 'depth.png')
            imgs.append(imageio.imread(fname))
            depths.append(imageio.imread(fname_d, pilmode='L'))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        depths = (np.array(depths) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_depths.append(depths)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(1)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        dpts_half_res = np.zeros((depths.shape[0], H, W))
        for i, (img, dpt) in enumerate(zip(imgs, depths)):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            dpts_half_res[i] = cv2.resize(dpt, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        depths = dpts_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, [H, W, focal], i_split, depths

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
    c2w[:,[1,2]] *= -1
    return c2w

def load_srn_data(basedir, half_res=False, sample_num=5, split='train'):
    depths = None
    pose_paths = sorted(glob.glob(os.path.join(basedir, 'pose', '*txt')))
    rgb_paths = sorted(glob.glob(os.path.join(basedir, 'rgb', '*png')))
    # coor_trans = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
    if sample_num > 0:
        sample_ind = random.sample(range(len(pose_paths)), sample_num)
        pose_paths = [pose_paths[i] for i in sample_ind]
        rgb_paths = [rgb_paths[i] for i in sample_ind]
    all_poses = []
    all_imgs = []
    i_split = [[], [], []]
    for i, (pose_path, rgb_path) in enumerate(zip(pose_paths, rgb_paths)):
        i_set = int(os.path.split(rgb_path)[-1][0])
        all_imgs.append((imageio.imread(rgb_path) / 255.).astype(np.float32))
        all_poses.append(np.loadtxt(pose_path).astype(np.float32).reshape(4, 4))
        i_split[i_set].append(i)

    imgs = np.stack(all_imgs, 0)
    poses = np.stack(all_poses, 0)

    H, W = imgs[0].shape[:2]
    with open(os.path.join(basedir, 'intrinsics.txt')) as f:
        focal = float(f.readline().split()[0])
    # R = np.sqrt((poses[...,:3,3]**2).sum(-1)).mean()
    # render_poses = torch.stack([pose_spherical(angle, -30.0, R) for angle in np.linspace(-180,180,200+1)[:-1]], 0)

    return imgs, poses, [H, W, focal], i_split, depths

def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
                      # it is only used to determined scene bbox
                      # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far

def load_data(datadir, half_res=False, sample_num=5, split='train',
              white_bkgd=False, load_depth=False, data_type='blender', load_mask=False):
    K, depths = None, None
    near_clip = None
    if data_type == 'blender':
        images, poses, hwf, i_split, depths = load_blender_data(datadir, half_res, sample_num, split=split)
    elif data_type == 'srn':
        images, poses, hwf, i_split, depths = load_srn_data(datadir, half_res, sample_num, split=split)
    else:
        raise NotImplementedError
    # print('Loaded blender', images.shape, hwf, datadir)
    i_train = i_val = i_test = i_split[0]
    if data_type == 'blender':
        near, far = 2., 4.
    elif data_type == 'srn':
        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])
    else:
        raise NotImplementedError

    # near, far = 2., 6.
    if load_mask:
        masks = images[..., -1]
    if images.shape[-1] == 4:
        if white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            # depths = depths[..., :3] * depths[..., -1:] + (1. - depths[..., -1:])
        else:
            images = images[..., :3] * images[..., -1:]
            # depths = depths[..., :3] * depths[..., -1:]
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks,
        near=near, far=far, #near_clip=near_clip,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses,
        images=images, #depths=depths,
        irregular_shape=irregular_shape,
    )
    if load_depth:
        data_dict['depths'] = depths
    if load_mask:
        data_dict['masks'] = masks
    return data_dict

def generalTransform(image, x_center, y_center, z_center, transform_matrix, method='linear'):
    # inverse matrix
    trans_mat_inv = np.linalg.inv(transform_matrix)
    # create coordinate meshgrid
    Nx, Ny, Nz = image.shape
    x = np.linspace(0, Nx - 1, Nx)
    y = np.linspace(0, Ny - 1, Ny)
    z = np.linspace(0, Nz - 1, Nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    # calculate transformed coordinate
    coor = np.array([xx - x_center, yy - y_center, zz - z_center])
    coor_prime = np.tensordot(trans_mat_inv, coor, axes=((1), (0)))
    xx_prime = coor_prime[0] + x_center
    yy_prime = coor_prime[1] + y_center
    zz_prime = coor_prime[2] + z_center
    # get valid coordinates (cell with coordinates within Nx-1, Ny-1, Nz-1)
    x_valid1 = xx_prime>=0
    x_valid2 = xx_prime<=Nx-1
    y_valid1 = yy_prime>=0
    y_valid2 = yy_prime<=Ny-1
    z_valid1 = zz_prime>=0
    z_valid2 = zz_prime<=Nz-1
    valid_voxel = x_valid1 * x_valid2 * y_valid1 * y_valid2 * z_valid1 * z_valid2
    x_valid_idx, y_valid_idx, z_valid_idx = np.where(valid_voxel > 0)
    # interpolate using scipy RegularGridInterpolator
    image_transformed = np.zeros((Nx, Ny, Nz))
    data_w_coor = RegularGridInterpolator((x, y, z), image, method=method)
    interp_points = np.array([xx_prime[x_valid_idx, y_valid_idx, z_valid_idx],
                                 yy_prime[x_valid_idx, y_valid_idx, z_valid_idx],
                                 zz_prime[x_valid_idx, y_valid_idx, z_valid_idx]]).T
    interp_result = data_w_coor(interp_points)
    image_transformed[x_valid_idx, y_valid_idx, z_valid_idx] = interp_result
    return image_transformed

def rodriguesRotate(image, x_center, y_center, z_center, axis, theta):
    v_length = np.linalg.norm(axis)
    if v_length==0:
        raise ValueError("length of rotation axis cannot be zero.")
    if theta==0.0:
        return image
    v = np.array(axis) / v_length
    # rodrigues rotation matrix
    W = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rot3d_mat = np.identity(3) + W * np.sin(theta) + np.dot(W, W) * (1.0 - np.cos(theta))
    # transform with given matrix
    return generalTransform(image, x_center, y_center, z_center, rot3d_mat, method='linear')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
def default_collate(batch):
    r"""
        Function that takes in a batch of data and puts the elements within the batch
        into a tensor with an additional outer dimension - batch size. The exact output type can be
        a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
        Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
        This is used as the default function for collation when
        `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

        Here is the general input type (based on the type of the element within the batch) to output type mapping:

            * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
            * NumPy Arrays -> :class:`torch.Tensor`
            * `float` -> :class:`torch.Tensor`
            * `int` -> :class:`torch.Tensor`
            * `str` -> `str` (unchanged)
            * `bytes` -> `bytes` (unchanged)
            * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
            * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]),
              default_collate([V2_1, V2_2, ...]), ...]`
            * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]),
              default_collate([V2_1, V2_2, ...]), ...]`

        Args:
            batch: a single batch to be collated

        Examples:
            >>> # Example with a batch of `int`s:
            >>> default_collate([0, 1, 2, 3])
            tensor([0, 1, 2, 3])
            >>> # Example with a batch of `str`s:
            >>> default_collate(['a', 'b', 'c'])
            ['a', 'b', 'c']
            >>> # Example with `Map` inside the batch:
            >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
            {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
            >>> # Example with `NamedTuple` inside the batch:
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_collate([Point(0, 0), Point(1, 1)])
            Point(x=tensor([0, 1]), y=tensor([0, 1]))
            >>> # Example with `Tuple` inside the batch:
            >>> default_collate([(0, 1), (2, 3)])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Example with `List` inside the batch:
            >>> default_collate([[0, 1], [2, 3]])
            [tensor([0, 2]), tensor([1, 3])]
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            try:
                storage = elem.storage()._new_shared(numel, device=elem.device)
            except:
                storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out).to(elem.device)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            out = {}
            for key in elem:
                if key == 'rgb_net_weight':
                    val = [d[key] for d in batch]
                elif  key == 'rotate_params':
                    val1 = [d[key][0] for d in batch]
                    val2 = [d[key][1] for d in batch]
                    val = (val1, val2)
                else:
                    val = default_collate([d[key] for d in batch])
                out[key] = val
            return out
            # return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [default_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


if __name__ == '__main__':
    # obj_list = os.listdir('/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/ShapeNet_Render/02876657')
    # obj_list2 = os.listdir('/media/huang/ZX3 512G/data/ShapeNet_Render/02876657')
    # dvgo_list = os.listdir('/media/huang/ZX3 512G/data/DVGO_results_64x64x64/02818832')
    # for item in obj_list:
    #     if item not in obj_list2:
    #         print(item)

    # tar_path = '/media/huang/T7/data/diff_nerf/DVGO_results_64x64x64'
    # image_path = '/media/huang/T7/data/diff_nerf/ShapeNet_Render'
    tar_path = '/media/huang/T7/data/abo_tables/DVGO_results_32x32x32'
    image_path = '/media/huang/T7/data/abo_tables/tables_train'
    from fvcore.common.config import CfgNode
    cfg = CfgNode({'white_bkgd': True,
                      'data_type': 'srn',})
    dataset = VolumeDataset(tar_path, image_path, use_rotate_transform=False, load_render_kwargs=True,
                            cfg=cfg, sample_num=-1)
    d = dataset[0]
    sd1 = 0
    sd2 = 0
    sd3 = 0
    maxm = -1e5
    minm = 1e5
    for i in range(len(dataset)):
        d = dataset[i]
        field = d['input']
        if field.max() > maxm:
            maxm = field.max()
        if field.min() < minm:
            minm = field.min()
        sd1 += field[:1, :].std()
        sd2 += field[1:, :].std()
        sd3 += field.std()
    dl = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,
                    pin_memory=True, num_workers=2, collate_fn=default_collate)
    def cycle(dl):
        while True:
            for data in dl:
                yield data
    # dl = cycle(dl)
    # while True:
    #     batch = next(dl)
    #     p = 0
    # angle = 90 / 180 * math.pi
    # # 创建一个坐标变换矩阵
    # transform_matrix = torch.tensor([
    #     [1, 0, 0, 0],
    #     [0, math.cos(angle), math.sin(-angle), 0],
    #     [0, math.sin(angle), math.cos(angle), 0],
    # ])
    # grid = F.affine_grid(transform_matrix.unsqueeze(0), density.unsqueeze(0).shape)
    # output = F.grid_sample(density.unsqueeze(0), grid, mode='nearest')[0]
    pause = 0