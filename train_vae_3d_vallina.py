# nerf optimizer and vae optimizer
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
# from diff_nerf.encoder_decoder_3d_2 import AutoencoderKL
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

class NeRF_Decoder(nn.Module):
    def __init__(self, cfg):
        super(NeRF_Decoder, self).__init__()
        viewbase_pe = cfg.render_kwargs.dvgo.viewbase_pe
        dim0 = (3 + 3 * viewbase_pe * 2)
        if cfg.render_kwargs.dvgo.rgbnet_full_implicit:
            pass
        elif cfg.render_kwargs.dvgo.rgbnet_direct:
            dim0 += cfg.render_kwargs.dvgo.rgbnet_dim
        else:
            dim0 += cfg.render_kwargs.dvgo.rgbnet_dim - 3
        rgbnet_width = cfg.render_kwargs.dvgo.rgbnet_width
        rgbnet_depth = cfg.render_kwargs.dvgo.rgbnet_depth
        self.residual = nn.Conv3d(4, 4, 1)
        nn.init.normal_(self.residual.weight, 0, 0.001)
        nn.init.constant_(self.residual.bias, 0)
        self.rgbnet = nn.Sequential(
            nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                for _ in range(rgbnet_depth - 2)
            ],
            nn.Linear(rgbnet_width, 3),
        )
        nn.init.constant_(self.rgbnet[-1].bias, 0)

def main(args):
    cfg = CfgNode(args.cfg)
    # logger = create_logger(root_dir=cfg['out_path'])
    # writer = SummaryWriter(cfg['out_path'])
    model_cfg = cfg.model
    model_kwargs = {'cfg': model_cfg}
    model_kwargs.update(model_cfg)
    model = construct_class_by_name(**model_kwargs)
    data_cfg = cfg.data
    data_kwargs = {'cfg': data_cfg}
    data_kwargs.update(data_cfg)
    dataset = construct_class_by_name(**data_kwargs)
    # dataset = VolumeDataset(tar_path=data_cfg.tar_path,
    #                         image_path=data_cfg.image_path,
    #                         load_rgb_net=data_cfg.get('load_rgb_net', False),
    #                         load_mask_cache=data_cfg.get('load_mask_cache', False),
    #                         use_rotate_ransform=data_cfg.get('use_rotate_ransform', False),
    #                         load_render_kwargs=data_cfg.get('load_render_kwargs', False),
    #                         sample_num=data_cfg.get("sample_num", 5),
    #                         normalize=data_cfg.get('normalize', False),
    #                         maxm=data_cfg.maxm, minm=data_cfg.minm, cfg=data_cfg)
    dl = DataLoader(dataset, batch_size=data_cfg.batch_size, shuffle=True,
                    pin_memory=True, num_workers=data_cfg.get("num_workers", 2), collate_fn=default_collate)
    train_cfg = cfg.trainer
    trainer = Trainer(
        model, dl, train_batch_size=data_cfg.batch_size,
        gradient_accumulate_every=train_cfg.gradient_accumulate_every,
        train_lr=train_cfg['lr'], train_num_steps=train_cfg.train_num_steps,
        save_and_sample_every=train_cfg.save_and_sample_every, results_folder=train_cfg.results_folder,
        amp=train_cfg['amp'], fp16=train_cfg.fp16, log_freq=train_cfg.log_freq, cfg=cfg,
        resume_milestone=train_cfg.get('resume_milestone', 0),
    )
    trainer.train()
    pass


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
        self.image_size = model.encoder.resolution

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
        # self.opt_ae = torch.optim.AdamW(param_dicts)
        self.opt_ae = torch.optim.AdamW(list(model.encoder.parameters())+
                                  list(model.decoder.parameters())+
                                  list(model.quant_conv.parameters())+
                                  list(model.post_quant_conv.parameters()),
                                  lr=train_lr)
        self.opt_disc = torch.optim.AdamW(model.loss.discriminator.parameters(), lr=train_lr)
        # self.opt_nerf = torch.optim.AdamW(list(model.rgbnet.parameters())+
        #                             list(model.residual.parameters()),
        #                             lr=5*train_lr)
        lr_lambda = lambda iter: max((1 - iter / train_num_steps) ** 0.95, cfg.trainer.min_lr)
        self.lr_scheduler_ae = torch.optim.lr_scheduler.LambdaLR(self.opt_ae, lr_lambda=lr_lambda)
        self.lr_scheduler_disc = torch.optim.lr_scheduler.LambdaLR(self.opt_disc, lr_lambda=lr_lambda)
        # self.lr_scheduler_nerf = torch.optim.lr_scheduler.LambdaLR(self.opt_nerf, lr_lambda=lr_lambda)
        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(model, ema_model=None, beta=0.998,
                           update_after_step=cfg.trainer.ema_update_after_step,
                           update_every=cfg.trainer.ema_update_every)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt_ae, self.opt_disc,  self.lr_scheduler_ae, self.lr_scheduler_disc = \
            self.accelerator.prepare(self.model, self.opt_ae, self.opt_disc, self.lr_scheduler_ae,
                                     self.lr_scheduler_disc)
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
            'opt_ae': self.opt_ae.state_dict(),
            'lr_scheduler_ae': self.lr_scheduler_ae.state_dict(),
            'opt_disc': self.opt_disc.state_dict(),
            'lr_scheduler_disc': self.lr_scheduler_disc.state_dict(),
            #'opt_nerf': self.opt_nerf.state_dict(),
            #'lr_scheduler_nerf': self.lr_scheduler_nerf.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        # msg = model.load_state_dict(data['model'])
        msg = model.load_state_dict(data['model'], strict=False)
        print('==>Resume AutoEncoder Info: ', msg)

        self.step = data['step']
        self.opt_ae.load_state_dict(data['opt_ae'])
        self.lr_scheduler_ae.load_state_dict(data['lr_scheduler_ae'])
        self.opt_disc.load_state_dict(data['opt_disc'])
        # self.opt_nerf.load_state_dict(data['opt_nerf'])
        self.lr_scheduler_disc.load_state_dict(data['lr_scheduler_disc'])
        # self.lr_scheduler_nerf.load_state_dict(data['lr_scheduler_nerf'])
        if self.accelerator.is_main_process:
            # self.ema.load_state_dict(data['ema'])
            self.ema.load_state_dict(data['ema'], strict=False)

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.
                batch = next(self.dl)
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key].to(device)
                for ga_ind in range(self.gradient_accumulate_every):
                    # with torch.no_grad():
                    #     if isinstance(self.model, nn.parallel.DistributedDataParallel):
                    #         rgbs, depths, bgmaps = self.model.module.render_img(batch, self.cfg.model.render_kwargs)
                    #     else:
                    #         rgbs, depths, bgmaps = self.model.render_img(batch, self.cfg.model.render_kwargs)
                    #     for rgb_i in range(len(rgbs)):
                    #         tv.utils.save_image(torch.from_numpy(rgbs[rgb_i]).permute(2, 0, 1),
                    #                             str(self.results_folder / f'ckpt-{4}-sample-{rgb_i}.png'))
                    #     exit()
                    with self.accelerator.autocast():
                        if isinstance(self.model, nn.parallel.DistributedDataParallel):
                            loss, log_dict = self.model.module.training_step(batch, ga_ind, self.step,
                                                    )
                        else:
                            loss, log_dict = self.model.training_step(batch, ga_ind, self.step,
                                                    )

                        # loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        if ga_ind == 0:
                            self.opt_ae.zero_grad()
                            self.opt_disc.zero_grad()
                            # if not self.cfg.trainer.get('skip_opt_net', False):
                            #     self.opt_nerf.zero_grad()
                            self.accelerator.backward(loss)
                            self.opt_ae.step()
                            # if not self.cfg.trainer.get('skip_opt_net', False):
                            #     self.opt_nerf.step()
                            rec_loss = log_dict["train/rec_loss"]
                            kl_loss = log_dict["train/kl_loss"]
                            d_weight = log_dict["train/d_weight"]
                            disc_factor = log_dict["train/disc_factor"]
                            g_loss = log_dict["train/g_loss"]
                        elif ga_ind == 1:
                            self.opt_disc.zero_grad()
                            self.accelerator.backward(loss)
                            self.opt_disc.step()
                            disc_loss = log_dict["train/disc_loss"]
                            logits_real = log_dict["train/logits_real"]
                            logits_fake = log_dict["train/logits_fake"]
                        else:
                            pass
                    # with self.accelerator.autocast():
                    #     if isinstance(self.model, nn.parallel.DistributedDataParallel):
                    #         render_lossdict = self.model.module.render_step(batch, self.accelerator,
                    #                                                                            self.opt_nerf)
                    #     else:
                    #         render_lossdict = self.model.render_step(batch, self.accelerator,
                    #                                                                     self.opt_nerf)
                    #     log_dict.update(render_lossdict)
                    # if accelerator.is_main_process:
                    #     pbar.desc = describtions
                    if self.step % self.log_freq == 0:
                        log_dict['lr'] = self.opt_ae.param_groups[0]['lr']
                        describtions = dict2str(log_dict)
                        describtions = "[Train Step] {}/{}: ".format(self.step, self.train_num_steps) + describtions
                        if accelerator.is_main_process:
                            pbar.desc = describtions
                            # self.logger.info(pbar.__str__())
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

                # self.opt_ae.step()
                # self.opt_ae.zero_grad()
                self.lr_scheduler_ae.step()
                # self.lr_scheduler_nerf.step()
                # self.opt_disc.step()
                # self.opt_disc.zero_grad()
                self.lr_scheduler_disc.step()
                if accelerator.is_main_process:
                    self.writer.add_scalar('Learning_Rate', self.opt_ae.param_groups[0]['lr'], self.step)
                    self.writer.add_scalar('total_loss', total_loss, self.step)
                    self.writer.add_scalar('rec_loss', rec_loss, self.step)
                    self.writer.add_scalar('kl_loss', kl_loss, self.step)
                    self.writer.add_scalar('d_weight', d_weight, self.step)
                    self.writer.add_scalar('disc_factor', disc_factor, self.step)
                    self.writer.add_scalar('g_loss', g_loss, self.step)
                    self.writer.add_scalar('disc_loss', disc_loss, self.step)
                    self.writer.add_scalar('logits_real', logits_real, self.step)
                    self.writer.add_scalar('logits_fake', logits_fake, self.step)

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step % 1000 == 0:
                        self.save('current')

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        #self.model.eval()
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
                        # with torch.no_grad():
                        #    if isinstance(self.model, nn.parallel.DistributedDataParallel):
                        #        rgbs, *_ = self.model.module.render_img(batch, self.cfg.model.render_kwargs)
                        #    else:
                        #        rgbs, *_ = self.model.render_img(batch, self.cfg.model.render_kwargs)
                        #    for rgb_i in range(len(rgbs)):
                        #        tv.utils.save_image(torch.from_numpy(rgbs[rgb_i]).permute(2, 0, 1),
                        #                            str(self.results_folder / f'ckpt-{milestone}-sample-{rgb_i}.png'))
                        # self.model.train()
                accelerator.wait_for_everyone()
                pbar.update(1)

        accelerator.print('training complete')


if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass