# Dual-Guidance Diffusion

A dual-guidance approach that leverages the pre-trained cross-modal model Contrastive Language-Image Pre-Training (CLIP) to guide the generation process. CLIP captures complex cross-modal semantics and styles, enhancing image-text alignment during sampling. Additionally, Cross-Attention Guidance (CAG) is proposed to further improve the image quality, which is a novel way that employs Gaussian blur to elim- inate redundant information in noisy intermediate images and utilizes cross-attention maps to highlight salient features. Remarkably, this dual-guidance is integrated into the sampling process without requiring additional training.

# Install

```
conda env create -f environment.yaml
conda activate dual
```
