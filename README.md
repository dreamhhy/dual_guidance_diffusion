# Dual-Guidance Diffusion

A dual-guidance approach that leverages the pre-trained cross-modal model Contrastive Language-Image Pre-Training (CLIP) to guide the generation process. CLIP captures complex cross-modal semantics and styles, enhancing image-text alignment during sampling. Additionally, Cross-Attention Guidance (CAG) is proposed to further improve the image quality, which is a novel way that employs Gaussian blur to elim- inate redundant information in noisy intermediate images and utilizes cross-attention maps to highlight salient features. Remarkably, this dual-guidance is integrated into the sampling process without requiring additional training.

# Install


```
git clone https://github.com/dreamhhy/dual_guidance_diffusion
cd dual_guidance_diffusion
pip install -e .

conda env create -f environment.yaml
conda activate dual
```


# Download Model Chekpoints

Stable Diffusion checkpoint
```
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
python split.py
```
CLIP image encoder checkpoint
```
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt -o /pretrained_models/clip/ViT-B-32.pt
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt -o /pretrained_models/clip/ViT-B-16.pt
wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt  -o /pretrained_models/clip/ViT-L-14.pt
wget https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt -o /pretrained_models/clip/RN50.pt
wget https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt  -o /pretrained_models/clip/RN50x4.pt
wget https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt -o /pretrained_models/clip/RN50x16.pt

```

# Sampling

```
python generate.py --prompt "A cyberpunk city with high architecture, wide angle, super highly detailed, professional digital painting"
```
