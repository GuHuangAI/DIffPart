import yaml
import argparse
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from diff_nerf.ema import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from diff_nerf.utils import *
import torchvision as tv
from diff_nerf.encoder_decoder3d import AutoencoderKL
# from denoising_diffusion_pytorch.data import ImageDataset, CIFAR10
from diff_nerf.data import VolumeDataset, default_collate
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from fvcore.common.config import CfgNode


def parse_args():
    parser = argparse.ArgumentParser(description="training vae configure")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, required=True)
    # parser.add_argument("")
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args

def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf

def main(args):
    cfg = CfgNode(args.cfg)
    model_cfg = cfg.model
    first_stage_cfg = model_cfg.first_stage
    first_stage_kwargs = {'cfg': first_stage_cfg}
    first_stage_kwargs.update(first_stage_cfg)
    first_stage_model = construct_class_by_name(**first_stage_kwargs)
    # if first_stage_cfg.type == 'ae3d_2':
    #     from diff_nerf.encoder_decoder_3d_2 import AutoencoderKL
    # elif first_stage_cfg.type == 'ae3d_3':
    #     from diff_nerf.encoder_decoder_3d_3 import AutoencoderKL
    # elif first_stage_cfg.type == 'ae3d_4':
    #     from diff_nerf.encoder_decoder_3d_4 import AutoencoderKL
    # elif first_stage_cfg.type == 'ae3d_5':
    #     from diff_nerf.encoder_decoder_3d_5 import AutoencoderKL
    # elif first_stage_cfg.type == 'ae3d_11':
    #     from diff_nerf.encoder_decoder_3d_11 import AutoencoderKL
    # else:
    #     raise NotImplementedError
    # first_stage_model = AutoencoderKL(
    #     ddconfig=first_stage_cfg.ddconfig,
    #     lossconfig=first_stage_cfg.lossconfig,
    #     embed_dim=first_stage_cfg.embed_dim,
    #     ckpt_path=first_stage_cfg.ckpt_path,
    #     cfg=first_stage_cfg,
    # )
    unet_cfg = model_cfg.unet
    unet = construct_class_by_name(**unet_cfg)
    # if model_cfg.model_name == '2dunet':
    #     unet_cfg = model_cfg.unet
    #     from diff_nerf.uncond_2dunet import create_model
    #     unet = create_model(unet_cfg)
    # elif model_cfg.model_name == '3dunet':
    #     unet_cfg = model_cfg.unet
    #     from diff_nerf.uncond_3dunet import Unet3D
    #     unet = Unet3D(
    #         dim=unet_cfg.dim,
    #         dim_mults=unet_cfg.dim_mults,
    #         channels=unet_cfg.in_channels,
    #         resnet_block_groups=unet_cfg.resnet_block_groups,
    #         heads=unet_cfg.heads,
    #         learned_sinusoidal_dim=unet_cfg.learned_sinusoidal_dim,
    #         window_size_q=unet_cfg.window_size_q,
    #         window_size_k=unet_cfg.window_size_k,
    #         out_mul=unet_cfg.out_mul,
    #     )
    # elif model_cfg.model_name == '3dunet_sb':
    #     unet_cfg = model_cfg.unet
    #     from diff_nerf.uncond_3dunet_sb import Unet3D
    #     unet = Unet3D(
    #         dim=unet_cfg.dim,
    #         dim_mults=unet_cfg.dim_mults,
    #         channels=unet_cfg.in_channels,
    #         resnet_block_groups=unet_cfg.resnet_block_groups,
    #         heads=unet_cfg.heads,
    #         learned_sinusoidal_dim=unet_cfg.learned_sinusoidal_dim,
    #         window_size_q=unet_cfg.window_size_q,
    #         window_size_k=unet_cfg.window_size_k,
    #         out_mul=unet_cfg.out_mul,
    #     )
    model_kwargs = {'model': unet, 'auto_encoder': first_stage_model, 'cfg': model_cfg}
    model_kwargs.update(model_cfg)
    ldm = construct_class_by_name(model_kwargs)
    # if model_cfg.model_type == 'const_sde':
    #     from diff_nerf.diff_nerf_const_sde_ldm import LatentDiffusion
    # elif model_cfg.model_type == 'const_sde_dis':
    #     from diff_nerf.diff_nerf_const_sde_ldm_distill import LatentDiffusion
    # else:
    #     raise NotImplementedError(f'{model_cfg.model_type} is not support !')
    # ldm = LatentDiffusion(
    #     model=unet,
    #     auto_encoder=first_stage_model,
    #     image_size=model_cfg.image_size,
    #     timesteps=model_cfg.timesteps,
    #     sampling_timesteps=model_cfg.sampling_timesteps,
    #     loss_type=model_cfg.loss_type,
    #     objective=model_cfg.objective,
    #     scale_factor=model_cfg.scale_factor,
    #     scale_by_std=model_cfg.scale_by_std,
    #     scale_by_softsign=model_cfg.scale_by_softsign,
    #     default_scale=model_cfg.get('default_scale', False),
    #     ckpt_path=model_cfg.ckpt_path,
    #     ignore_keys=model_cfg.ignore_keys,
    #     only_model=model_cfg.only_model,
    #     start_dist=model_cfg.start_dist,
    #     use_l1=model_cfg.get('use_l1', True),
    #     cfg=model_cfg,
    # )
    data_cfg = cfg.data
    data_kwargs = {'cfg': data_cfg}
    data_kwargs.update(data_cfg)
    dataset = construct_class_by_name(**data_kwargs)
    dl = DataLoader(dataset, batch_size=data_cfg.batch_size, shuffle=True,
                    pin_memory=True, num_workers=2, collate_fn=default_collate)
    train_cfg = cfg.trainer
    trainer = Trainer(
        ldm, dl, train_batch_size=data_cfg.batch_size,
        gradient_accumulate_every=train_cfg.gradient_accumulate_every,
        train_lr=train_cfg['lr'], train_num_steps=train_cfg.train_num_steps,
        save_and_sample_every=train_cfg.save_and_sample_every, results_folder=train_cfg.results_folder,
        amp=train_cfg['amp'], fp16=train_cfg.fp16, log_freq=train_cfg.log_freq, cfg=cfg,
        resume_milestone=train_cfg.get('resume_milestone', 0),
    )
    if train_cfg.test_before:
        if trainer.accelerator.is_main_process:
            with torch.no_grad():
                for datatmp in dl:
                    break
                for key in datatmp.keys():
                    if isinstance(datatmp[key], torch.Tensor):
                        datatmp[key] = datatmp[key].to(trainer.accelerator.device)
                        # print(trainer.accelerator.device)
                if isinstance(trainer.model, nn.parallel.DistributedDataParallel):
                    rgbs, depths, bgmaps = trainer.model.module.render_img(datatmp, first_stage_cfg.render_kwargs)
                elif isinstance(trainer.model, nn.Module):
                    rgbs, depths, bgmaps = trainer.model.render_img(datatmp, first_stage_cfg.render_kwargs)
                # all_images = torch.clamp((all_images + 1.0) / 2.0, min=0.0, max=1.0)

            # all_images = torch.cat(all_images_list, dim = 0)
            for rgb_i in range(len(rgbs)):
                tv.utils.save_image(torch.from_numpy(rgbs[rgb_i]).permute(2, 0, 1),
                                    str(trainer.results_folder / f'ckpt-{train_cfg.resume_milestone}-testbefore-{rgb_i}.png'))
            # nrow = 2 ** math.floor(math.log2(math.sqrt(data_cfg.batch_size)))
            # tv.utils.save_image(all_images, str(trainer.results_folder / f'sample-{train_cfg.resume_milestone}_{model_cfg.sampling_timesteps}.png'), nrow=nrow)
            torch.cuda.empty_cache()
    trainer.train()

class Trainer(object):
    def __init__(
        self,
        model,
        data_loader,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        log_freq = 10,
        resume_milestone = 0,
        cfg={}
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no',
            kwargs_handlers=[ddp_handler],
        )

        self.accelerator.native_amp = amp
        self.cfg = cfg
        self.model = model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.log_freq = log_freq

        self.train_num_steps = train_num_steps
        # self.image_size = model.encoder.resolution

        # dataset and dataloader

        # self.ds = Dataset(folder, mask_folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        # dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(data_loader)
        self.dl = cycle(dl)

        # optimizer
        # param_dicts = [
        #     {'params': list(model.encoder.parameters())+
        #                           list(model.decoder.parameters())+
        #                           list(model.quant_conv.parameters())+
        #                           list(model.post_quant_conv.parameters()),
        #      'lr': train_lr},
        #     {'params': list(model.rgbnet.parameters())+list(model.residual.parameters()),
        #      'lr': train_lr*5}
        # ]

        self.opt_d = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.model.parameters()),
                                     lr=train_lr, weight_decay=cfg.trainer.train_wd)
        self.opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.first_stage_model.parameters()),
                                     lr=train_lr*0.1, weight_decay=cfg.trainer.train_wd)
        lr_lambda = lambda iter: max((1 - iter / train_num_steps) ** 0.95, cfg.trainer.min_lr)
        self.lr_scheduler_d = torch.optim.lr_scheduler.LambdaLR(self.opt_d, lr_lambda=lr_lambda)
        self.lr_scheduler_v = torch.optim.lr_scheduler.LambdaLR(self.opt_v, lr_lambda=lr_lambda)
        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(model, ema_model=None, beta=0.999,
                           update_after_step=cfg.trainer.ema_update_after_step,
                           update_every=cfg.trainer.ema_update_every)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt_d, self.opt_v,  self.lr_scheduler_d, self.lr_scheduler_v = \
            self.accelerator.prepare(self.model, self.opt_d, self.opt_v, self.lr_scheduler_d,
                                     self.lr_scheduler_v)
        self.logger = create_logger(root_dir=results_folder)
        self.logger.info(cfg)
        self.cfg = cfg
        self.writer = SummaryWriter(results_folder)
        self.results_folder = Path(results_folder)
        resume_file = str(self.results_folder / f'model-{resume_milestone}.pt')
        if os.path.isfile(resume_file):
            self.load(resume_milestone)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt_d': self.opt_d.state_dict(),
            'lr_scheduler_d': self.lr_scheduler_d.state_dict(),
            'opt_v': self.opt_v.state_dict(),
            'lr_scheduler_v': self.lr_scheduler_v.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'),
                          map_location=lambda storage, loc: storage)

        model = self.accelerator.unwrap_model(self.model)
        msg = model.load_state_dict(data['model'])
        print('==>Resume Model Info: ', msg)
        if 'scale_factor' in data['model']:
            model.scale_factor = data['model']['scale_factor']

        self.step = data['step']
        self.opt_d.load_state_dict(data['opt_d'])
        self.lr_scheduler_d.load_state_dict(data['lr_scheduler_d'])
        self.opt_v.load_state_dict(data['opt_v'])
        self.lr_scheduler_v.load_state_dict(data['lr_scheduler_v'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.
                total_loss_dict = {'loss_simple': 0., 'loss_vlb': 0., 'loss_render': 0., 'loss_cls': 0.,
                                   'total_loss': 0., 'psnr': 0., 'acc': 0., 'lr': 5e-5,
                                   }
                for ga_ind in range(self.gradient_accumulate_every):
                    # data = next(self.dl).to(device)
                    batch = next(self.dl)
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            batch[key].to(device)
                    if self.step == 0 and ga_ind == 0:
                        if isinstance(self.model, nn.parallel.DistributedDataParallel):
                            self.model.module.on_train_batch_start(batch)
                        else:
                            self.model.on_train_batch_start(batch)

                    with self.accelerator.autocast():
                        if isinstance(self.model, nn.parallel.DistributedDataParallel):
                            loss, log_dict = self.model.module.training_step(batch, global_step=self.step,
                                                                             accelerator=self.accelerator,
                                                                             opt_nerf=self.opt_v
                                                                             )
                        else:
                            loss, log_dict = self.model.training_step(batch, global_step=self.step,
                                                                      accelerator=self.accelerator,
                                                                      opt_nerf=self.opt_v
                                                                      )

                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                        loss_simple = log_dict["train/loss_simple"].item()
                        loss_vlb = log_dict["train/loss_vlb"].item()
                        loss_render = log_dict["loss_render"] if 'loss_render' in log_dict else 0
                        psnr = log_dict["psnr"] if 'psnr' in log_dict else 0
                        acc = log_dict['acc'] if 'acc' in log_dict else 0
                        total_loss_dict['loss_simple'] += loss_simple
                        total_loss_dict['loss_vlb'] += loss_vlb
                        total_loss_dict['loss_render'] += loss_render
                        total_loss_dict['total_loss'] += total_loss
                        total_loss_dict['psnr'] += psnr
                        # total_loss_dict['s_fact'] = self.model.module.scale_factor
                        # total_loss_dict['s_bias'] = self.model.module.scale_bias

                    self.accelerator.backward(loss)
                log_dict['lr'] = self.opt_d.param_groups[0]['lr']
                describtions = dict2str(log_dict)
                describtions = "[Train Step] {}/{}: ".format(self.step, self.train_num_steps) + describtions
                if accelerator.is_main_process:
                    pbar.desc = describtions
                if self.step % self.log_freq == 0:
                    if accelerator.is_main_process:
                        # pbar.desc = describtions
                        self.logger.info(describtions)
                '''
                with tqdm(initial=1, total=self.cfg.model.render_kwargs.inner_iter,
                          disable=not accelerator.is_main_process) as pbar2:
                    for inner_iter in range(self.cfg.model.render_kwargs.inner_iter):
                        with self.accelerator.autocast():
                            if isinstance(self.model, nn.parallel.DistributedDataParallel):
                                render_loss, render_lossdict, psnr = self.model.module.render_step(batch, self.accelerator, self.opt_nerf)
                            else:
                                render_loss, render_lossdict, psnr = self.model.render_step(batch, self.accelerator, self.opt_nerf)
                            # log_dict.update(render_lossdict)
                            self.opt_nerf.zero_grad()
                            self.accelerator.backward(render_loss)
                            self.opt_nerf.step()
                            pbar2.set_postfix({
                                'inner_iter': inner_iter,
                                'render_loss': render_loss,
                                'psnr': '%.3f' % psnr,
                            })
                        pbar2.update(1)
                '''
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                # pbar.set_description(f'loss: {total_loss:.4f}')
                accelerator.wait_for_everyone()

                self.opt_d.step()
                self.opt_d.zero_grad()
                self.lr_scheduler_d.step()
                self.opt_v.step()
                self.opt_v.zero_grad()
                self.lr_scheduler_v.step()
                if accelerator.is_main_process:
                    self.writer.add_scalar('Learning_Rate', self.opt_d.param_groups[0]['lr'], self.step)
                    self.writer.add_scalar('total_loss', total_loss, self.step)
                    self.writer.add_scalar('loss_simple', loss_simple, self.step)
                    self.writer.add_scalar('loss_vlb', loss_vlb, self.step)
                    self.writer.add_scalar('loss_render', loss_render, self.step)
                    self.writer.add_scalar('psnr', psnr, self.step)
                    self.writer.add_scalar('acc', acc, self.step)

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step % 1000 == 0:
                        self.save('current')

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.model.eval()
                        '''
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            # img = self.dl
                            #batches = num_to_groups(self.num_samples, self.batch_size)
                            #all_images_list = list(map(lambda n: self.model.module.validate_img(ns=self.batch_size), batches))
                            if isinstance(self.model, nn.parallel.DistributedDataParallel):
                                all_images = self.model.module.validate_img(img[:2])
                            elif isinstance(self.model, nn.Module):
                                all_images = self.model.validate_img(img[:2])
                            all_images = torch.clamp((all_images + 1.0) / 2.0, min=0.0, max=1.0)

                        nrow = 2 ** math.floor(math.log2(math.sqrt(self.batch_size)))
                        tv.utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = nrow)
                        '''
                        milestone = self.step // self.save_and_sample_every
                        self.save(milestone)
                        with torch.no_grad():
                            if isinstance(self.model, nn.parallel.DistributedDataParallel):
                                rgbs, depths, bgmaps = self.model.module.render_img(batch, self.cfg.model.first_stage.render_kwargs)
                            else:
                                rgbs, depths, bgmaps = self.model.render_img(batch, self.cfg.model.first_stage.render_kwargs)
                            for rgb_i in range(len(rgbs)):
                                tv.utils.save_image(torch.from_numpy(rgbs[rgb_i]).permute(2, 0, 1),
                                                    str(self.results_folder / f'ckpt-{milestone}-sample-{rgb_i}.png'))
                        self.model.train()
                accelerator.wait_for_everyone()
                pbar.update(1)

        accelerator.print('training complete')


if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass