from einops import rearrange
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch

from img2img import load_model_from_config, load_img

if __name__ == "__main__":
    model_class = 'kl-f8'
    verbose = True

    device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")

    config = OmegaConf.load('models/first_stage_models/'+model_class+'/config.yaml')
    ckpt = 'models/first_stage_models/'+model_class+'/model.ckpt'
    model = load_model_from_config(config, ckpt, verbose=verbose)
    model = model.to(device)

    image_path = '/dreambig/qingyi/image_chicago/data/images/satellite/zoom17/17_31_10100_3.png'
    image = Image.open(image_path).convert("RGB")

    image = image.crop((172,172,428,428))
    image = np.array(image).astype(np.float32) / 255.0
    # image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2.*image - 1.
    image = image.to(device)

    # latent = model.encode(img)
    # reconstruct = model.decode(latent)
    reconstruct, _ = model.forward(image)#, sample_posterior=False)
    reconstruct = torch.clamp((reconstruct + 1.0) / 2.0, min=0.0, max=1.0)
    reconstruct = 255. * rearrange(reconstruct.cpu().detach().numpy()[0,:,:,:], 'c h w -> h w c')

    Image.fromarray(reconstruct.astype(np.uint8)).save('AE_test/sample_'+model_class+'-31_10100_3.png')