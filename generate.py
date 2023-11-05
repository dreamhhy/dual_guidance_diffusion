import os
import gc
import io
import math
import sys

from PIL import Image, ImageOps, ImageDraw
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from torchvision.ops import masks_to_boxes
import importlib
from tqdm.notebook import tqdm

import numpy as np

from rich import print
from rich.align import Align
from rich.panel import Panel
from omegaconf import OmegaConf
from einops import rearrange
from math import log2, sqrt

import argparse
import pickle

from utils import set_requires_grad, fetch, parse_prompt, range_loss, tv_loss, spherical_dist_loss, MakeCutouts
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, classifier_defaults, create_classifier
from clip_custom import clip
from transformers import CLIPTokenizer, CLIPTextModel, BertModel, BertTokenizer
sys.path.append('./latent_diffusion')
from ldm.util import instantiate_from_config



# Args
def parse_arguments():
    parser = argparse.ArgumentParser(description='Image generation using CLIP + classifier free diffusion')
    
    # prompt
    parser.add_argument('--prompt', 
                        type = str, 
                        required = False,
                        help='input a prompt', 
                        default = "Legendary fairy prince casting lightning spell, blue energy, fantasy, high detail, digital painting, concept art.")
    
    parser.add_argument('--negative', 
                        type = str, 
                        required = False, 
                        default = '',
                        help='negative text prompt')
    
    # model path
    parser.add_argument('--model_path', 
                        type=str, 
                        default = 'diffusion.pt', 
                        help='path of the diffusion model')
    
    parser.add_argument('--autoencoder_path', 
                        type=str, 
                        default = 'kl.pt', 
                        help='path of the autoencoder model')
    
    parser.add_argument('--autoencoder_config_path', 
                        type=str, 
                        default = 'kl.yaml',
                        help='path of the autoencoder config')
    
    parser.add_argument('--clip_model', 
                        type=str, 
                        choices=("ViT-L-14", "ViT-B-16", "ViT-B-32", "RN50", "RN50x4", "RN50x16", "RN50x64", "RN101"),
                        default = 'ViT-B-32', 
                        help='choose clip model')
    
    parser.add_argument('--clip_path', 
                        type=str, 
                        default = '/g/data/jr19/hh4436/my/cfg_diffusion/pretrained_models/clip/', 
                        help='path of the clip model')
    
    # generate setttings
    parser.add_argument('--CLIP_Guidance', type = bool, default = True, help='If use CLIP guidance')
    parser.add_argument('--CAG', type = bool, default = True, help='If use CAG')
    
    parser.add_argument('--blur_sigma', type = int, default = 9, help='The value of blur sigma')
    parser.add_argument('--init_image', type=str, required = False, default = None, help='init image to use')
    parser.add_argument('--skip_timesteps', type=int, required = False, default = 0, help='how many diffusion steps skipped')
    
    parser.add_argument('--width', type = int, default = 512, required = False, help='image size of output (multiple of 8)')
    parser.add_argument('--height', type = int, default = 512, required = False, help='image size of output (multiple of 8)')
    parser.add_argument('--seed', type = int, default=42, required = False, help='random seed')
    parser.add_argument('--cfg_scale', type = float, default = 8.0, required = False, help='classifier-free guidance scale')
    parser.add_argument('--clip_guidance_scale', type = float, default = 150, required = False, help='clip guidance scale')
    parser.add_argument('--timestep_respacing', type = str, default = "", required = False, help='number of diffusion steps')
    
    parser.add_argument('--cutn', type = int, default = 16, help='Number of cuts')
    parser.add_argument('--tv_scale', type=float, default=120, help="Smoothness scale")
    parser.add_argument('--range_scale', type=int, default=50, help="RGB range scale")
    parser.add_argument('--num_cuts', type=int, default=16, help="Number of cuts")
    parser.add_argument('--num_cut_batches', type=int, default=2, help="Number of cut batches")
    parser.add_argument('--cut_power', type=float, default=1., help="Cut power")
    
    # sampler
    parser.add_argument('--ddpm', type = bool, default = False, help='choose to use ddpm sampler')
    parser.add_argument('--ddim', type = bool, default = True, help='choose to use ddim sampler')
    parser.add_argument('--plms', type = bool, default = False, help='choose to use plms sampler')
    parser.add_argument('--steps', type = str, default = '50', help='timestep_respacing, the number of sample steps')
    
    # save
    parser.add_argument('--prefix', type = str, required = False, default = '', help='prefix for output files')
    parser.add_argument('--num_batches', type = int, default = 6, required = False, help='number of batches')
    parser.add_argument('--batch_size', type = int, default = 1, required = False, help='batch size')
    
    args = parser.parse_args()
    
    return args



if __name__ == '__main__':
    
    print(Panel(Align("Dual-Guidance Diffusion", "center")))
    
    args = parse_arguments()
    
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    print(f'Using device: "{device}"')
    
    
    # seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f'Seed: {args.seed}')
    else:
        seed = torch.seed()
        torch.manual_seed(seed)
        print(f'Seed: {seed}')
    
    # image size
    print(f'Image size: width {args.width} x height {args.height}')
    
    # Set DDPM
    model_state_dict = torch.load(args.model_path, map_location='cpu')
    
    model_params = {
    'attention_resolutions': '32,16,8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '50',
    'image_size': 32,
    'learn_sigma': False,
    'noise_schedule': 'linear',
    'num_channels': 320,
    'num_heads': 8,
    'num_res_blocks': 2,
    'resblock_updown': False,
    'use_fp16': False,
    'use_scale_shift_norm': False,
    'clip_embed_dim': None,
    'image_condition': True if model_state_dict['input_blocks.0.0.weight'].shape[1] == 8 else False,
    'super_res_condition': True if 'external_block.0.0.weight' in model_state_dict else False,
    'sel_attn_depth': 3,
    'sel_attn_block': "output"
    }
    
    if device == 'cpu':
        model_config['use_fp16'] = False
    
    if args.ddpm:
        model_params['timestep_respacing'] = '1000'
        print(f'Use DDPM sampler, steps: 1000') 
    elif args.ddim:
        if args.steps:
            model_params['timestep_respacing'] = 'ddim'+str(args.steps)
            print(f'Use DDIM sampler, steps: {args.steps}')
        else:
            model_params['timestep_respacing'] = 'ddim250'
            print(f'Use DDIM sampler, steps: 250')
    elif args.plms:
        model_params['timestep_respacing'] = str(args.steps)
        print(f'Use PLMS sampler, steps: {args.steps}')
        
    model_config = model_and_diffusion_defaults()
    model_config.update(model_params)
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(model_state_dict)
    model.requires_grad_(True).eval().to(device)
    
    if model_config['use_fp16']:
        model.convert_to_fp16()
    else:
        model.convert_to_fp32()
        
    if args.ddpm:
        sample_fn = diffusion.ddpm_sample_loop_progressive
    elif args.ddim:
        sample_fn = diffusion.ddim_sample_loop_progressive
    elif args.plms:
        sample_fn = diffusion.plms_sample_loop_progressive

        
    print(f'Finish loading DDPM')    
    
    
    # Load VAE
    kl_config = OmegaConf.load(args.autoencoder_config_path)
    kl_sd = torch.load(args.autoencoder_path, map_location="cpu")

    vae = instantiate_from_config(kl_config.model)
    vae.load_state_dict(kl_sd, strict=True)
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    set_requires_grad(vae, False)
    
    print(f'Finish loading VAE')
    
    # Load text encoder
    # clip
    clip_version = 'openai/clip-vit-large-patch14'
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_version)
    clip_transformer = CLIPTextModel.from_pretrained(clip_version)
    clip_transformer.eval().requires_grad_(False).to(device)
    
    print(f'Finish loading text encoder')
    
    # load CLIP model
    clip_dir = args.clip_path + args.clip_model + '.pt'
    clip_model, clip_preprocess = clip.load(clip_dir, device=device, jit=False)
    clip_model.eval().requires_grad_(False)
    set_requires_grad(clip_model, False)
    clip_size = clip_model.visual.input_resolution
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    
    print(f'Finish loading CLIP, model: {args.clip_model}')
    
    
    ###############
    #     Run     #
    ###############
    
    # Text encoder
    text = clip_tokenizer([args.prompt]*args.batch_size, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    text_blank = clip_tokenizer([args.negative]*args.batch_size, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    text_tokens = text["input_ids"].to(device)
    text_blank_tokens = text_blank["input_ids"].to(device)

    text_emb = clip_transformer(input_ids=text_tokens).last_hidden_state
    text_emb_blank = clip_transformer(input_ids=text_blank_tokens).last_hidden_state
    

    image_embed = None
    
    init = None
    
    kwargs = {
        "context": torch.cat([text_emb, text_emb_blank], dim=0).float(),
        "clip_embed": None,
        "image_embed": image_embed
    }
    
    make_cutouts = MakeCutouts(clip_size, args.num_cuts, args.cut_power)
    
    
    # Add attention
    text_attn = clip.tokenize([args.prompt]*args.batch_size, truncate=True).to(device)
    text_clip_emb = clip_model.encode_text(text_attn)

    query = key = value = text_clip_emb
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
    text_clip_emb_attn = torch.matmul(attention_weights, value)
    
    
    # some parameters used in diffusion process
    linear_start = 0.00085
    linear_end = 0.012
    betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, 1000, dtype=np.float64) ** 2
    betas = np.array(betas, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
    log_one_minus_alphas_cumprod = np.log(1.0 - alphas_cumprod)
    

    def _extract_into_tensor(arr, timesteps, broadcast_shape):
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    
    def _predict_xstart_from_eps(x_t, t, eps):
            assert x_t.shape == eps.shape
            return (
                _extract_into_tensor(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
            )

    def q_sample(x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def attention_masking(
        x0, x, t, attn_map, prev_noise, blur_sigma, model_kwargs=None,
    ):
        """
        Apply the cross-attention mask to produce bar{x_t}
        
        :param x_0: the predicted x_0 at time t
        :param x: the current noisy x at time t.
        :param t: a 1-D Tensor of timesteps.
        :param attn_map: the attention map tensor at time t.
        :param prev_noise: the previously predicted epsilon to inject the same noise as x_t.
        :param blur_sigma: a sigma of Gaussian blur.
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.
        :return: the bar{x_t}
        """
        B, C, H, W = x.shape
        assert t.shape == (B,)
        
        # Generating attention mask
        attn_mask = attn_map.reshape(B, 4, 80, 64, 64).mean(dim=2).sum(1, keepdim=False) > 1.0 
        attn_mask = attn_mask.unsqueeze(1).repeat(1, 4, 1, 1).int().float()

        # Gaussian blur
        transform = T.GaussianBlur(kernel_size=31, sigma=3)
        x_curr = transform(x0)
        x_curr = q_sample(x_curr, t, noise=prev_noise)

        # Apply attention masking
        x_curr = x_curr * (attn_mask) * 1.5 + x
        
        return x_curr
    
    
    cur_t = None
    
    
    # Create a classifier-free based (CAG) guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        
        """
        Do the classififer-free guidance based job
 
        :param x_t: the current noisy x at time t.
        :param ts: a 1-D Tensor of timesteps.
        :return: the predicted noise
        """
        
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        
        model_out = model(combined, ts, **kwargs)
        
        # Get the attention map from the U-Net
        attn = model.output_blocks[10][1].att
            
        n = x_t.shape[0]
        my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
        
        # Get the predicted x_0
        x0 = _predict_xstart_from_eps(x_t, my_t, model_out)
        
        # Do mask and return new x_t
        mask_blurred = attention_masking(
                x0,
                x_t,
                my_t,
                attn,
                prev_noise=model_out,
                blur_sigma=args.blur_sigma,
            )
        
        # Get new eps
        uncond_att_eps = model(mask_blurred, ts, **kwargs)
        eps_att, rest_att = uncond_att_eps[:, :3], uncond_att_eps[:, 3:]
        cond_eps_att, uncond_eps_att = torch.split(eps_att, len(eps_att) // 2, dim=0)

        # The eps from the original classifier-free 
        eps, rest = model_out[:, :3], model_out[:, 3:]

        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    
        if args.CAG:
            guided_eps = uncond_eps + args.cfg_scale * (cond_eps_att - uncond_eps)
        else:
            guided_eps = uncond_eps + args.cfg_scale * (cond_eps - uncond_eps)

        eps = torch.cat([guided_eps, guided_eps], dim=0)

        return torch.cat([eps, rest], dim=1)
    
    
    cond_fn = None
    

    def cond_fn(x, t, context=None, clip_embed=None, image_embed=None):
        """
        Do CLIP guidance
 
        :param x: the current noisy x at time t.
        :param t: a 1-D Tensor of timesteps.
        :return: the gradient of the distance loss between generated image and prompt
        """
  
        with torch.enable_grad():

            x = x[:args.batch_size].detach().requires_grad_()
            n = x.shape[0]

            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
            
            kw = {'context': context[:args.batch_size], 
                  'clip_embed': clip_embed[:args.batch_size] if model_params['clip_embed_dim'] else None,
                  'image_embed': image_embed[:args.batch_size] if image_embed is not None else None
            }

            out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs=kw)
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]

            x_in = out['pred_xstart'] * fac + x * (1 - fac)
            x_in /= 0.18215
            x_in = vae.decode(x_in)
            
            clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
            clip_embeds = clip_model.encode_image(clip_in).float()
            
            dists = spherical_dist_loss(clip_embeds.unsqueeze(1), text_clip_emb_attn.unsqueeze(0))
            dists = dists.view([args.cutn, n, -1])
            
            losses = dists.sum(2).mean(0)
            tv_losses = tv_loss(x_in)
            # range_losses = range_loss(out['pred_xstart'])
            loss = losses.sum() * args.clip_guidance_scale + tv_losses.sum() * args.tv_scale
            gradient = -torch.autograd.grad(loss, x)[0]
 
            return gradient
    
    # Create output folders
    os.makedirs("output", exist_ok = True)
    os.makedirs("output_npy", exist_ok = True)
    
    def save_sample(i, samples, square=None):
        for k, image in enumerate(samples):
            image_scaled = image/0.18215
            im = image_scaled.unsqueeze(0)
            out = vae.decode(im)

            npy_filename = f'output_npy/{args.prefix}{i * args.batch_size + k:05}.npy'
            with open(npy_filename, 'wb') as outfile:
                np.save(outfile, image_scaled.detach().cpu().numpy())

            out = TF.to_pil_image(out.squeeze(0).add(1).div(2).clamp(0, 1))
            
            if square is not None:
                outdraw = ImageDraw.Draw(out)  
                outdraw.rectangle([(square[0]*8, square[1]*8),(square[0]*8+512, square[1]*8+512)], fill=None, outline ="red")

            filename = f'output/{args.prefix}{i * args.batch_size + k:05}.png'
            out.save(filename)
            
    for i in range(args.num_batches):
            cur_t = diffusion.num_timesteps - 1

            samples = sample_fn(
                model_fn,
                (args.batch_size*2, 4, int(args.height/8), int(args.width/8)),
                clip_denoised=False,
                model_kwargs=kwargs,
                cond_fn=cond_fn,
                device=device,
                progress=True,
                init_image=init,
                skip_timesteps=args.skip_timesteps,
            )

            for j, sample in enumerate(samples):
                cur_t -= 1
                if j % 20 == 0:
                    save_sample(i, sample['pred_xstart'][:args.batch_size])

            save_sample(i, sample['pred_xstart'][:args.batch_size])

    gc.collect()
    