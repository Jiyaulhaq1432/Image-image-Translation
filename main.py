import torch
from diffusers import StableDiffusionImg2ImgPipeline

class CustomModel:
    def __init__(self, base_model):
        self.base_model = base_model

    def __call__(self, image, prompt):
        return self.base_model(prompt=prompt, image=image, strength=0.75, guidance_scale=7.5).images[0]

def load_pretrained_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    return pipe

if __name__ == '__main__':
    pretrained_model = load_pretrained_model()
    model = CustomModel(pretrained_model)
    print("Model loaded and ready for inference")