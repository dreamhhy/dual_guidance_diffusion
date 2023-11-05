# Dual-Guidance Diffusion

A dual-guidance approach that leverages the pre-trained cross-modal model Contrastive Language-Image Pre-Training (CLIP) to guide the generation process. CLIP captures complex cross-modal semantics and styles, enhancing image-text alignment during sampling. Additionally, Cross-Attention Guidance (CAG) is proposed to further improve the image quality, which is a novel way that employs Gaussian blur to elim- inate redundant information in noisy intermediate images and utilizes cross-attention maps to highlight salient features. Remarkably, this dual-guidance is integrated into the sampling process without requiring additional training.

# Install



```
git clone [https://github.com/Jack000/glid-3](https://github.com/dreamhhy/dual_guidance_diffusion
cd dual_guidance_diffusion
pip install -e .

conda env create -f environment.yaml
conda activate dual
```
# Download Model Chekpoints

```
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
python split.py
```

# Sampling

```
python generate.py
```
