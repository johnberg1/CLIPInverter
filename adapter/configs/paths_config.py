dataset_paths = {
    # Face Datasets
    'celeba_train': '',
    'celeba_test': '',
    'ffhq': '',

    # Animal Datasets
    'afhq_train': '',
    'afhq_test': '',
    'cub_train': '',
    'cub_test': ''
}

model_paths = {
    'pretrained_e4e_encoder': 'pretrained_models/e4e_ffhq_encode.pt',
    'ir_se50': 'pretrained_models/model_ir_se50.pth',
    'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'shape_predictor': 'pretrained_models/shape_predictor_68_face_landmarks.dat',
    'moco': 'pretrained_models/moco_v2_800ep_pretrain.pt'
}

caption_paths = {
    'celeba_caption_train': '',
    'celeba_caption_test': ''
}
