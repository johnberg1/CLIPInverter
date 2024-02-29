from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--no_coarse_mapper', default=False, action="store_true" , help='Whether to use coarse feature adapter')
		self.parser.add_argument('--no_medium_mapper', default=False, action="store_true", help='Whether to use medium feature adapter')
		self.parser.add_argument('--no_fine_mapper', default=False, action="store_true", help='Whether to use fine feature adapter')
		self.parser.add_argument('--use_moco', default=False, action="store_true", help='Whether to use MoCo loss')
		self.parser.add_argument('--dataset_type', default='celeba_encode', type=str, help='Type of dataset/experiment to run')

		self.parser.add_argument('--start_from_latent_avg', action='store_true', help='Whether to add average latent vector to generate codes from encoder.')

		self.parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')

		self.parser.add_argument('--learning_rate', default=0.0005, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')

		self.parser.add_argument('--lpips_lambda', default=0, type=float, help='LPIPS loss multiplier factor')
		self.parser.add_argument('--l2_lambda', default=0, type=float, help='L2 loss multiplier factor')
		self.parser.add_argument('--w_norm_lambda', default=0, type=float, help='W-norm loss multiplier factor')
        
		self.parser.add_argument('--id_lambda', default=0, type=float, help='ID loss multiplier factor (ArcFace and MOCO)')
		self.parser.add_argument('--clip_lambda', default=1.0, type=float, help='clip loss multiplier factor')
		self.parser.add_argument('--cycle_lambda', default=0, type=float, help='Cycle loss multiplier factor')
        
		self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--stylegan_size', default=1024, type=int)
		self.parser.add_argument('--ir_se50_weights', default=model_paths['ir_se50'], type=str, help="Path to facial recognition network used in ID loss")
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to CLIPInverter model checkpoint')
		self.parser.add_argument('--pretrained_e4e_path', default=model_paths['pretrained_e4e_encoder'], type=str, help='Path to e4e weights')

		self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=2000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=2000, type=int, help='Model checkpoint interval')

	def parse(self):
		opts = self.parser.parse_args()
		return opts