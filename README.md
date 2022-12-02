# cog-stable-diffusion-inpainting-v2


[![Replicate](https://replicate.com/replicate/stable-diffusion-inpainting/badge)](https://replicate.com/replicate/stable-diffusion-inpainting) 

This is an implementation of the [Diffusers Stable Diffusion v2](https://huggingface.co/stabilityai/stable-diffusion-2) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i prompt="..." -i image=@... -i mask=@...
