# CLIPInverter: CLIP-Guided StyleGAN Inversion for Text-driven Real Image Editing (ACM TOG 2023)

<a href="https://arxiv.org/abs/2307.08397"><img src="https://img.shields.io/badge/arXiv-2307.08397-b31b1b.svg"></a> <a href="https://dl.acm.org/doi/10.1145/3610287"><img src="https://img.shields.io/badge/ACM_TOG-CLIPInverter-maroon"></a> <a href="https://cyberiada.github.io/CLIPInverter/"><img src="https://img.shields.io/badge/Project_Page-purple"></a>


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

**29.02.2024**: Training code is released. Please note that the pretrained model checkpoint for CLIPInverter is updated with this commit together with some minor changes to the inference script `infer.py` and the model definitions. If you use the previous checkpoint with the new scripts, you might run into some errors or unexpected results. Please download the newest checkpoint from the link down below.

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
|[CLIPInverter Faces](https://drive.google.com/file/d/18goTnPtVrz1Tuen3JuDIEwj5z3GvgVqJ/view?usp=sharing) | CLIPInverter trained on CelebA-HQ. Includes the StyleGAN2 weights too.
|[Dlib alignment](https://drive.google.com/file/d/1uoOsJcT0bC-_zNDbhcj6iaxLJBN-LFao/view?usp=sharing) | Dlib alignment used for images preproccessing.
|[FFHQ e4e encoder](https://drive.google.com/file/d/1kxYtrg4YQCudxL5f9xmCzOdJRITH5UXB/view?usp=share_link) | Pretrained e4e encoder
|[StyleGAN2 FFHQ](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view) | StyleGAN model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.


By default, we assume that all models are downloaded and saved to the directory `pretrained_models`. However, you may use your own paths by changing the necessary values in `configs/path_configs.py`. 

## Training
To train CLIPInverter, make sure the paths to the required models, as well as training and testing data is configured in `configs/path_configs.py` and `configs/data_configs.py`.

### Training on your own dataset
In order to train CLIPInverter on a custom dataset, perform the following adjustments:
1. Insert the paths to your train and test data into the `dataset_paths` and `caption_paths` variables defined in `configs/paths_config.py`. `dataset_paths` should contain the image folders and `caption_paths` should contain the caption folders:
```
dataset_paths = {
    'my_train_data': '/path/to/train/images/directory',
    'my_test_data': '/path/to/test/images/directory'
}

caption_paths = {
    'my_data_captions_train': '/path/to/train/captions/directory',
    'my_data_captions_test': '/path/to/test/captions/directory'
}
```
2. Configure a new dataset under the DATASETS variable defined in `configs/data_configs.py`:
```
DATASETS = {
   'my_data_encode': {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['my_train_data'],
        'train_target_root': dataset_paths['my_train_data'],
        'test_source_root': dataset_paths['my_test_data'],
        'test_target_root': dataset_paths['my_test_data']
    }
}
```
Refer to `configs/transforms_config.py` for the transformations applied to the train and test images during training. 

3. Finally, run a training session with `--dataset_type my_data_encode`.

**Important Note**: The dataset class `ImagesTextDataset` in `datasets/images_text_dataset.py` is defined for the CelebA-HQ dataset, where there are 10 captions corresponding to each image. The captions corresponding to an image `idx.jpg` is included in a text file named `idx.txt` with the same index. We randomly sample one of the 10 captions during training to increase diversity of the captions. If your dataset has a different structure, you may need to define a new dataset class and use that class definition for training.

### Running Training Scripts
We follow a two-stage training regime, where we train CLIPAdapter at the first stage and train CLIPRemapper at the second stage.

#### First Stage Training
We use the following arguments for the first stage training. Please check `options/train_options.py` for a full list of arguments. Note that by default, we use ArcFace for the ID loss, which works for human face datasets. If your dataset includes animals, you need to include the flag `--use_moco` in the training arguments. The argument `--id_lambda` is used for the weight of either ArcFace or MoCo losses.
```
python scripts/train_first_stage.py \
--dataset_type celeba_encode \
--exp_dir=new/experiment/directory \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=10000 \
--save_interval=10000 \
--id_lambda=0.1 \
--cycle_lambda=1 \
--lpips_lambda=0.6 \
--l2_lambda=1 \
--w_norm_lambda=0.005 \
--stylegan_size=1024 \
--max_steps=200000 \
--stylegan_weights=path/to/pretrained/stylegan.pt \
--start_from_latent_avg \
```

#### Second Stage Training
We use the following arguments for the first stage training. Please check `options/train_options_stage_two.py` for a full list of arguments. Please make sure to include the first stage trained model with the argument `--checkpoint_path`
```
python scripts/train_second_stage.py \
--dataset_type celeba_encode \
--exp_dir=new/experiment/directory  \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=10000 \
--save_interval=10000 \
--id_lambda=0.15 \
--cycle_lambda=1 \
--lpips_lambda=0.15 \
--l2_lambda=0.15 \
--clip_lambda=1.75 \
--learning_rate=0.00015 \
--w_norm_lambda=0.005 \
--stylegan_size=1024 \
--max_steps=20000 \
--stylegan_weights=path/to/pretrained/stylegan.pt \
--start_from_latent_avg \
--checkpoint_path=path/to/stage/one/model.pt \
--is_training_from_stage_one
```

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