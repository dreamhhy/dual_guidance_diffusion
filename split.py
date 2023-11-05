import torch
import sys

model_path = 'sd-v1-4.ckpt'

state = torch.load(model_path)['state_dict']

model = {}
vae = {}

model_prefix = 'model.diffusion_model.'
vae_prefix = 'first_stage_model.'

for key in state.keys():
    if key.startswith(model_prefix):
        model[key[len(model_prefix):]] = state[key]
    elif key.startswith(vae_prefix):
        vae[key[len(vae_prefix):]] = state[key]

torch.save(model, 'diffusion.pt')
torch.save(vae, 'kl.pt')
