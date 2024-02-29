import os
import random
import clip
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common
from criteria import id_loss, w_norm, moco_loss
from configs import data_configs
from datasets.images_text_dataset import ImagesTextDataset
from criteria.lpips.lpips import LPIPS
from criteria.clip_loss import DirectionalCLIPLoss

from adapter.adapter_first_stage import CLIPAdapterStyleGAN
from adapter.training.ranger import Ranger
from adapter.training import train_utils
from models.e4e_features import pSp

class Coach:
	def __init__(self, opts):
		self.opts = opts
		self.global_step = 0
		self.device = 'cuda:0'
		self.opts.device = self.device
		self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device = self.device)

		# Initialize network
		self.net = CLIPAdapterStyleGAN(self.opts).to(self.device)
		self.encoder = pSp(self.opts).to(self.device).eval()
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
	
		# Initialize loss
		if not self.opts.use_moco:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()
		else:
			self.id_loss = moco_loss.MocoLoss().to(self.device).eval()
		self.mse_loss = nn.MSELoss().to(self.device).eval()
		self.directional_loss = DirectionalCLIPLoss(self.clip_model)
		self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		self.w_norm_loss = w_norm.WNormLoss(opts=self.opts)        
		
		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)



		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.log_dir = log_dir
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				x, y, txt = batch
				x, y = x.to(self.device).float(), y.to(self.device).float()
				text_original = clip.tokenize(txt).to(self.device)
				with torch.no_grad():
					txt_embed_original = self.clip_model.encode_text(text_original)
				txt_embed_original = txt_embed_original.to(self.device).float()
				self.optimizer.zero_grad()
                
				# We use a matching text with the input image 25% of the time
				mismatch_text = random.random() <= (3. / 4)
				if mismatch_text:
					txt_embed_mismatch = torch.roll(txt_embed_original, 1, dims=0)
					text_mismatch = torch.roll(text_original, 1, dims=0)
				else:
					txt_embed_mismatch = txt_embed_original
					text_mismatch = text_original
                    
                # Forward pass
				with torch.no_grad():
					w, features = self.encoder.forward(x, return_latents=True)
                
				features = self.net.adapter(features, txt_embed_mismatch)
				w_hat = w + 0.1 * self.encoder.forward_features(features)
				y_hat, w_hat = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
				y_hat = self.face_pool(y_hat)
				loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, text_original, text_mismatch, w_hat, y, mismatch_text, data_type="real")
				loss.backward()
                
                # Cycle pass
				y_hat_clone = y_hat.clone().detach().requires_grad_(True)
				txt_embed_clone = txt_embed_original.clone().detach().requires_grad_(True)
				text_clone = text_original
                
				with torch.no_grad():
					w_cycle, features_cycle = self.encoder.forward(y_hat_clone, return_latents=True)
				features_cycle = self.net.adapter(features_cycle, txt_embed_clone)
				w_hat_cycle = w_cycle + 0.1 * self.encoder.forward_features(features_cycle)
				y_recovered, w_hat_cycle = self.net.decoder([w_hat_cycle], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
				y_recovered = self.face_pool(y_recovered)
				loss, cycle_loss_dict, cycle_id_logs = self.calc_loss(x, y, y_recovered, text_mismatch, text_clone, w_hat_cycle, y_hat_clone, mismatch_text, data_type="cycle")
				loss.backward()
				self.optimizer.step()
                
                # combine the logs of both forwards
				for idx, cycle_log in enumerate(cycle_id_logs):
					id_logs[idx].update(cycle_log)
				loss_dict.update(cycle_loss_dict)
				loss_dict["loss"] = loss_dict["loss_real"] + loss_dict["loss_cycle"]
                
                # Logging related
				if self.global_step % self.opts.image_interval == 0 or \
						(self.global_step < 1000 and self.global_step % 25 == 0):
					self.parse_and_log_images(id_logs, x, y, y_hat, y_recovered, txt, mismatch_text,
											  title='images/train/faces')

				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!', flush=True)
					break

				self.global_step += 1
                
	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
            
			x, y, txt = batch
			text_original = clip.tokenize(txt).to(self.device)
			with torch.no_grad():
				txt_embed_original = self.clip_model.encode_text(text_original)
				txt_embed_original = txt_embed_original.to(self.device).float()
				x, y = x.to(self.device).float(), y.to(self.device).float()

				# Always mismatching text when validating
				mismatch_text = True
				txt_embed_mismatch = torch.roll(txt_embed_original, 1, dims=0)
				text_mismatch = torch.roll(text_original, 1, dims=0)

                    
                # Forward pass
				w, features = self.encoder.forward(x, return_latents=True)
				features = self.net.adapter(features, txt_embed_mismatch)
				w_hat = w + 0.1 * self.encoder.forward_features(features)
				y_hat, w_hat = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
				y_hat = self.face_pool(y_hat)
				loss, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, text_original, text_mismatch, w_hat, y, mismatch_text)
                
                # Cycle pass
				y_hat_clone = y_hat.clone().detach().requires_grad_(True)
				txt_embed_clone = txt_embed_original.clone().detach().requires_grad_(True)
				text_clone = text_original
                
				w_cycle, features_cycle = self.encoder.forward(y_hat_clone, return_latents=True)
				features_cycle = self.net.adapter(features_cycle, txt_embed_clone)
				w_hat_cycle = w_cycle + 0.1 * self.encoder.forward_features(features_cycle)
				y_recovered, w_hat_cycle = self.net.decoder([w_hat_cycle], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
				y_recovered = self.face_pool(y_recovered)
				loss, cycle_loss_dict, cycle_id_logs = self.calc_loss(x, y, y_recovered, text_mismatch, text_clone, w_hat_cycle, y_hat_clone, mismatch_text, data_type="cycle")
                
				# combine the logs of both forwards
				for idx, cycle_log in enumerate(cycle_id_logs):
					id_logs[idx].update(cycle_log)
				cur_loss_dict.update(cycle_loss_dict)
				cur_loss_dict["loss"] = cur_loss_dict["loss_real"] + cur_loss_dict["loss_cycle"]
                
			agg_loss_dict.append(cur_loss_dict)

			
			# Logging related
			self.parse_and_log_images(id_logs, x, y, y_hat, y_recovered, txt, mismatch_text, title='images/test/faces',
									  subscript='{:04d}'.format(batch_idx))

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else 'latest_model.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

	def configure_optimizers(self):
		params = list(self.net.adapter.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self):
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		train_dataset = ImagesTextDataset(source_root=dataset_args['train_source_root'],
									  target_root=dataset_args['train_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts, train=True)
		test_dataset = ImagesTextDataset(source_root=dataset_args['test_source_root'],
									 target_root=dataset_args['test_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts, train=False)
			
		print("Number of training samples: {}".format(len(train_dataset)), flush=True)
		print("Number of test samples: {}".format(len(test_dataset)), flush=True)
		return train_dataset, test_dataset

	def calc_loss(self, x, y, y_hat, source_text, target_text, latent, directional_source, mismatch_text, data_type="real"):
		loss_dict = {}
		id_logs = []
		loss = 0.0
		if self.opts.id_lambda > 0:
			if not self.opts.use_moco:
				loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x, label=None, weights=None)
				loss_dict[f'loss_id_{data_type}'] = float(loss_id)
				loss_dict[f'id_improve_{data_type}'] = float(sim_improvement)
			else:
				loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
				loss_dict[f'loss_id_{data_type}'] = float(loss_id)
				loss_dict[f'id_improve_{data_type}'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda
            
		if self.opts.l2_lambda > 0:
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict[f'loss_l2_{data_type}'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda
            
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict[f'loss_lpips_{data_type}'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda
            
		if self.opts.w_norm_lambda > 0:
			loss_w_norm = self.w_norm_loss(latent, latent_avg=self.encoder.latent_avg)
			loss_dict[f'loss_w_norm_{data_type}'] = float(loss_w_norm)
			loss += loss_w_norm * self.opts.w_norm_lambda
		
		# CLIP loss
		if mismatch_text and self.opts.clip_lambda > 0: 
			loss_directional = self.directional_loss(directional_source, y_hat, source_text, target_text).mean()
			loss_dict[f'loss_directional_{data_type}'] = float(loss_directional)
			loss += loss_directional * self.opts.clip_lambda   
		loss_dict[f'loss_{data_type}'] = float(loss)
		if data_type == "cycle":
			loss = loss * self.opts.cycle_lambda
		return loss, loss_dict, id_logs
    
	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	def parse_and_log_images(self, id_logs, x, y, y_hat, y_recovered, txt, mismatch_text, title, subscript=None, display_count=2):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.tensor2im(x[i]),
				'output_face': common.tensor2im(y_hat[i]),
                'recovered_face': common.tensor2im(y_recovered[i])
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, txt, mismatch_text, im_data=im_data, subscript=subscript)		
        
	def log_images(self, name, txt, mismatch_text, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data, txt, mismatch_text)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
		else:
			path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		# # save the latent avg in state_dict for inference if truncation of w was used during training
		# if self.encoder.latent_avg is not None:
		# 	save_dict['latent_avg'] = self.encoder.latent_avg
		return save_dict