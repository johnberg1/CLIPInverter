# CLIPInverter: CLIP-Guided StyleGAN Inversion for Text-driven Real Image Editing (ACM TOG 2023)

<a href="https://arxiv.org/abs/2307.08397"><img src="https://img.shields.io/badge/arXiv-2008.00951-b31b1b.svg"></a> <a href="https://dl.acm.org/doi/10.1145/3610287"><img src="https://img.shields.io/badge/ACM_TOG-CLIPInverter-maroon"></a> <a href="https://cyberiada.github.io/CLIPInverter/"><img src="https://img.shields.io/badge/Project_Page-purple"></a>


Inference Notebook: <a target="_blank" href="https://colab.research.google.com/github/johnberg1/CLIPInverter/blob/main/CLIPInverter_Inference.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

>Official Implementation of CLIP-Guided StyleGAN Inversion for Text-Driven Real Image Editing published in ACM TOG 2023 and presented in SIGGRAPH ASIA 2023 Syndey.

<p align="center">
<img src="assets/teaser_image_v3.jpg"/>  
<br>
We present CLIPInverter that enables users to easily perform semantic changes on images using free natural text. Our approach is not specific to a certain category of images and can be applied to many different domains (e.g., human faces, cats, birds) where a pretrained StyleGAN generator exists (top). Our approach specifically gives more accurate results for multi-attribute edits as compared to the prior work (middle). Moreover, as we utilize CLIP’s semantic embedding space, it can also perform manipulations based on reference images without any training or finetuning (bottom).
</br>
</p>

## Updates
**27.07.2023**: Our HuggingFace demo is released.

**29.08.2023**: CLIPInverter is published in ACM TOG!

**06.12.2023**: Inference code is released.

**SOON**: Training code will be released soon.

## HuggingFace Demo
You can use your own images and captions with CLIPInverter using our demo: [![HuggingFace](https://img.shields.io/badge/HuggingFace-CLIPInverter-blue)](https://huggingface.co/spaces/johnberg/CLIPInverter)

## Getting Started
### Prerequisites

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

### Pretrained Models

Pretrained models are available on the following links. 

| Path | Description
| :--- | :----------
|[CLIPInverter Faces](https://drive.google.com/file/d/1GKVAgd8g3_ckc3ZiApYLHEcKmxrisZ9_/view?usp=share_link) | CLIPInverter trained on CelebA-HQ. Includes the StyleGAN2 weights too.
|[Dlib alignment](https://drive.google.com/file/d/1uoOsJcT0bC-_zNDbhcj6iaxLJBN-LFao/view?usp=sharing) | Dlib alignment used for images preproccessing.
|[FFHQ e4e encoder](https://drive.google.com/file/d/1kxYtrg4YQCudxL5f9xmCzOdJRITH5UXB/view?usp=share_link) | Pretrained e4e encoder

By default, we assume that all models are downloaded and saved to the directory `pretrained_models`. 

## Testing
### Inference

You may use `infer.py` to evaluate the models. The results will be saved in a directory named `results`. The script has 4 arguments and the usage is as follows:

```bash
python infer.py \
--input_image_path=/path/to/input/image \
--caption="target description" \
--model_path=/path/to/clipinverter/model \
--e4e_path=/path/to/pretrained/e4e/ \
```

#### Example Usage
```bash
python infer.py \
--input_image_path=assets/sample_input.jpg \
--caption="this person has a beard" \
--model_path=pretrained_models/pretrained_faces.pt \
--e4e_path=pretrained_models/e4e_ffhq_encode.pt \
```

## Acknowledgements
This repository is based on [StyleGAN2 (Rosinality)](https://github.com/rosinality/stylegan2-pytorch),  [encoder4editing](https://github.com/omertov/encoder4editing) and [HairCLIP](https://github.com/wty-ustc/HairCLIP).

## Citation

```
@article{baykal2023clipinverter,
author = {Baykal, Ahmet Canberk and Anees, Abdul Basit and Ceylan, Duygu and Erdem, Erkut and Erdem, Aykut and Yuret, Deniz},
title = {CLIP-Guided StyleGAN Inversion for Text-Driven Real Image Editing},
year = {2023},
issue_date = {October 2023},
publisher = {Association for Computing Machinery},
volume = {42},
number = {5},
journal = {ACM Trans. Graph.}
```