# Decoupled Diffusion Models with Explicit Transition Probability

## I. Before Starting.
1. install torch
~~~
torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
~~~
2. install other packages.
~~~
pip install -r requirement.txt
~~~
3. prepare accelerate config.
~~~
accelerate config
~~~

## II. Prepare Data.
The file structure should look like:
```commandline
data_root
|-- images
|   |-- XXX.jpg
|   |-- XXX.jpg
|-- condition_images
|   |-- XXX.jpg
|   |-- XXX.jpg
```

## III. Unconditional training on image space for Cifar10 dataset.
~~~
accelerate launch train_uncond_dpm.py --cfg ./configs/cifar10/uncond_const_dpm_sde4_ncsnpp9.yaml
~~~

## IV. Unconditional training on latent space for CelebAHQ256 dataset.
1. training auto-encoder:
~~~
accelerate launch train_vae.py --cfg ./configs/celebahq/celeb_ae_kl_256x256_d4.yaml
~~~
2. you should add the model weights in the first step to config file `./configs/inpainting/celebahq_256x256_ldm_etp_const_sde4.yaml`, then train latent diffusion model:
~~~
accelerate launch train_uncond_ldm.py --cfg ./configs/celebahq/uncond_etp_const_ldm_sde.yaml
~~~

## V. Conditional training on latent space for CelebAHQ256 dataset. (Inpainting task for example.)
~~~
accelerate launch train_cond_ldm.py --cfg ./configs/inpainting/celebahq_256x256_ldm_etp_const_sde4.yaml
~~~

## VI. Faster Sampling
**change the sampling steps "sampling_timesteps" in the config file**
1. unconditional generation:
~~~
python sample_uncond.py --cfg ./configs/cifar10/uncond_const_dpm_sde4_ncsnpp9.yaml
~~~
2. conditional generation (Latent space model):
~~~
python sample_cond_ldm.py --cfg ./configs/inpainting/celebahq_256x256_ldm_etp_const_sde4.yaml
~~~