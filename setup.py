from setuptools import setup

setup(
    name="dual-guidance-diffusion",
    py_modules=["dual_guidance_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)