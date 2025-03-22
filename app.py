import os
import torch
import streamlit as st
from PIL import Image
from diffusers import RePaintPipeline, RePaintScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mask_image = Image.open('./mask.png').convert("RGB").resize((256, 256))
example_image = Image.open('./example.png').convert("RGB").resize((256, 256))

scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
pipe = RePaintPipeline.from_pretrained("google/ddpm-ema-celebahq-256", scheduler=scheduler)
pipe = pipe.to(device)
generator = torch.Generator(device=device).manual_seed(0)

def main():
    st.title('Image Inpainting using Denoising Diffusion Probabilistic Model')
    st.subheader('Model: DDPM - Repaint. Dataset: CelebA-HQ')
    option = st.selectbox('How would you like to give the input?', ('Upload Image File', 'Run Example Image'))
    if option == "Upload Image File":
        file = st.file_uploader("Please upload an image", type=["jpg", "png"])
        if file is not None:
            image = Image.open(file).convert("RGB").resize((256, 256))
            output = pipe(
                image=image,
                mask_image=mask_image,
                num_inference_steps=10,
                generator=generator,
            )
            st.image(output.images[0])
          
    elif option == "Run Example Image":
        output = pipe(
            image=example_image,
            mask_image=mask_image,
            num_inference_steps=10,
            generator=generator,
        )
        st.image(output.images[0])

if __name__ == '__main__':
    main() 
