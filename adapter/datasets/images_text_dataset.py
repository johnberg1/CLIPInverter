from torch.utils.data import Dataset
from configs.paths_config import caption_paths
from PIL import Image
from utils import data_utils
import random
import os
from io import BytesIO
import torchvision.transforms as transforms

class ImagesTextDataset(Dataset):

    def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None, train=True):
        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.opts = opts
        self.train = train
        if train:
            self.text_paths_dir = caption_paths['celeba_caption_train']
        else:
            self.text_paths_dir = caption_paths['celeba_caption_test']
        self.text_paths = sorted(data_utils.make_text_dataset(self.text_paths_dir))
        self.rsz_transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __len__(self):
        #if self.train and self.opts is not None and self.opts.datasetSize is not None and len(self.source_paths) >= self.opts.datasetSize:
            #return self.opts.datasetSize
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = Image.open(from_path)
        from_im = from_im.convert('RGB')

        to_path = self.target_paths[index]
        to_im = Image.open(to_path).convert('RGB')
        orig = self.rsz_transform(from_im)
        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)
        else:
            from_im = to_im
      
        txt_file_dir = self.text_paths[index]
        txt_file = open(txt_file_dir,"r")
        txt = txt_file.read().splitlines()
        txt = random.choice(txt) # randomly choosing a caption among 10 captions for each image

        return from_im, to_im, txt
  


class CatsDataset(Dataset):

    def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None, train=True):
        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.opts = opts
        self.train = train
        if train:
            self.text_paths_dir = caption_paths['afhq_caption_train']
        else:
            self.text_paths_dir = caption_paths['afhq_caption_test']
        self.text_paths = sorted(data_utils.make_text_dataset(self.text_paths_dir))
        self.rsz_transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __len__(self):
        #if self.train and self.opts is not None and self.opts.datasetSize is not None and len(self.source_paths) >= self.opts.datasetSize:
            #return self.opts.datasetSize
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = Image.open(from_path)
        from_im = from_im.convert('RGB')

        to_path = self.target_paths[index]
        to_im = Image.open(to_path).convert('RGB')
        orig = self.rsz_transform(from_im)
        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)
        else:
            from_im = to_im
      
        txt_file_dir = self.text_paths[index]
        txt_file = open(txt_file_dir,"r")
        txt = txt_file.read().splitlines()
        txt = random.choice(txt) 

        return from_im, to_im, txt

class LMDBDataset(Dataset):
	def __init__(self, source_root, resolution, filenames=None, imgfolder=None, textfolder=None, transform = None):
		import lmbd
		self.source_root = source_root
		self.source_paths = None
		self.text_paths = None
		if filenames:
			self.source_paths, self.text_paths = self.getfiles(filenames, imgfolder, textfolder)
		self.env = lmdb.open(source_root,max_readers=32,readonly=True,lock=False,readahead=False,meminit=False)
		if not self.env:
			raise IOError('Cannot open lmdb dataset', source_root)
		with self.env.begin(write=False) as txn:
			self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
		self.resolution = resolution
		self.transform = transform

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		with self.env.begin(write=False) as txn:
			key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
			img_bytes = txn.get(key)
		buffer = BytesIO(img_bytes)
		img = Image.open(buffer)
		img = self.transform(img)
		to_im = img
		from_im = to_im
		txt_file_dir = self.text_paths[index]
		txt_file = open(txt_file_dir,"r")
		txt = txt_file.read().splitlines()
		txt = random.choice(txt) #txt[0]
		txt_file.close()
		return from_im, to_im, txt

	def getfiles(self, filenames, img_folder, text_folder):
		imgs = []
		assert os.path.isfile(filenames)
		i = 0

		f = open(filenames, "r")
		for line in f:
			if i % 2 != 0:
				fields = line.rstrip().split(' ')
				imgs.append(fields[0])
			if i % 2 != 0:
				if i > 1:
					imgs[i//2] = imgs[i//2][2:]
				else:
					imgs[i//2] = imgs[i//2][1:]
			i += 1

		imgs.pop()
		f.close()
		images = [os.path.join(img_folder, image_path+'.jpg') for image_path in imgs]
		texts = [os.path.join(text_folder, image_path+'.txt') for image_path in imgs]
		return imgs, texts