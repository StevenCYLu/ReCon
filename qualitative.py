from diffusers import StableDiffusionXLPipeline
from PIL import Image
import numpy as np
import torch

import pickle
from datasets import load_dataset
from transformers import CLIPTextModel, CLIPTokenizer
import faiss

import spacy

nlp = spacy.load("en_core_web_trf")
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    cache_dir=".cache",
)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

index = faiss.read_index("faiss_index_concept_diffdb.index")
with open('metadata_dict_concept_diffdb.pkl', 'rb') as f:
    metadata_dict = pickle.load(f)
pipeline.to(device)

output_dir = "experiments/qualitative/steps"


def decode(pipeline, latents):
    needs_upcasting = (
        pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast
    )

    if needs_upcasting:
        pipeline.upcast_vae()
        latents = latents.to(
            next(iter(pipeline.vae.post_quant_conv.parameters())).dtype
        )

    image = pipeline.vae.decode(
        latents / pipeline.vae.config.scaling_factor, return_dict=False
    )[0].detach()

    # cast back to fp16 if needed
    if needs_upcasting:
        pipeline.vae.to(dtype=torch.float16)
    image = pipeline.image_processor.postprocess(image, output_type="pil")
    return image

def main():

    prompt = 'close-up photography of an old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux'
    
    doc = nlp(prompt)
    cache_prompts = [chunk.text for chunk in doc.noun_chunks]
    # cache_prompts = ['an old man', 'the rain at night', 'street lamps']

    inputs = tokenizer(cache_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        text_features = outputs.last_hidden_state.mean(dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    query = text_features.cpu().numpy().astype('float32')
    D, I = index.search(query, 1)
    
    cache_prompts = []
    
    for ind in range(len(I)):
        ind = I[ind][0]
        cache_prompts.append(metadata_dict[ind]['caption'])
    cache_prompts = cache_prompts[:-1]

    for k in range(5, 31, 5):
        cache_latents = []
        for ind, cache in enumerate(cache_prompts):
            latents = []

            def callback(step, t, latent, nuff):
                if step % 5 == 0:
                    latents.append(latent)

            cache_img = pipeline(
                cache,
                callback=callback,
                negative_prompt="anime, vector art, art, painting, illustration, ugly, abstract, black and white",
                width=768,
                height=768,
                generator=torch.Generator("cuda").manual_seed(1337),
                num_inference_steps = 50,
                cross_attention_kwargs={},
            )
            cache_latents.append(latents[k // 5])
            cache_img.images[0].save(f"{output_dir}/{cache}.png")
        cache_latents = torch.mean(torch.stack(cache_latents), dim=0)
        latents = []
        vis = []
        def callback(step, t, latent, noise_pred):
            # print(noise_pred.shape)
            if step % 5 == 0:
                # latents.append(latent)
                vis.append(noise_pred)
        output = pipeline(
            prompt,
            latents=cache_latents,
            callback=callback,
            width=768,
            height=768,
            negative_prompt="anime, vector art, art, painting, illustration, ugly, abstract, black and white",
            cross_attention_kwargs={},
            start_ratio=(1 + k) / 50,
            num_inference_steps = 50,
            generator=torch.Generator("cuda").manual_seed(1337),
        ).images[0]
        output.save(f"{output_dir}/concept_k={k}.png")
        

if __name__ == "__main__":
    main()
