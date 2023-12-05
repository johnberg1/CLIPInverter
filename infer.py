import argparse

import torch
from argparse import Namespace
import torchvision.transforms as transforms
import clip
import numpy as np
import sys
sys.path.append(".")
sys.path.append("..")
from models.e4e_features import pSp
from adapter.adapter_decoder import CLIPAdapterWithDecoder
from PIL import Image

def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return var.astype('uint8')

def run_alignment(image_path):
    import dlib
    from align_faces_parallel import align_face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(image_path, predictor=predictor)
    # print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

def load_model(model_path, e4e_path):
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts['pretrained_e4e_path'] = e4e_path
    opts['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    opts = Namespace(**opts)
    encoder = pSp(opts)
    encoder.eval()
    encoder.cuda()

    adapter = CLIPAdapterWithDecoder(opts)
    adapter.eval()
    adapter.cuda()

    clip_model, _ = clip.load("ViT-B/32", device='cuda')
    return encoder, adapter, clip_model

def manipulate(input_image_path, caption, encoder, adapter, clip_model):
    input_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    input_image = Image.open(input_image_path).convert('RGB')
    aligned_image = run_alignment(input_image)
    input_image = input_transforms(aligned_image)
    input_image = input_image.unsqueeze(0)
    text_input = clip.tokenize(caption)
    text_input = text_input.cuda()
    input_image = input_image.cuda().float()

    with torch.no_grad():
        text_features = clip_model.encode_text(text_input).float()

        w, features = encoder.forward(input_image, return_latents=True)
        features = adapter.adapter(features, text_features)
        w_hat = w + 0.1 * encoder.forward_features(features)
        
        result_tensor, _ = adapter.decoder([w_hat], input_is_latent=True, return_latents=False, randomize_noise=False, truncation=1, txt_embed=text_features)
        result_tensor = result_tensor.squeeze(0)
        result_image = tensor2im(result_tensor)
        result_image = Image.fromarray(result_image)

    return result_image

def main(args):
    encoder, adapter, clip_model = load_model(args.model_path, args.e4e_path)
    result_image = manipulate(args.input_image_path, args.caption, encoder, adapter, clip_model)
    result_image.save("result.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_path", type=str, required=True)
    parser.add_argument("--caption", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--e4e_path", type=str, required=True)
    args = parser.parse_args()
    main(args)

