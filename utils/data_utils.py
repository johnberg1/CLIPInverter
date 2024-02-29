"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

TXT_EXTENSIONS = [
    '.txt'
]

def is_txt_file(filename):
    return any(filename.endswith(extension) for extension in TXT_EXTENSIONS)
    
def make_text_dataset(dir):
    texts = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_txt_file(fname):
                path = os.path.join(root, fname)
                texts.append(path)
    return texts


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), f'{dir} is not a valid directory'
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def make_dataset_from_paths_list(paths_file):
    assert os.path.exists(paths_file), f'{paths_file} is not a valid file'
    with open(paths_file, "r") as f:
        paths = f.readlines()
    paths = [p.strip() for p in paths]
    paths = [p for p in paths if is_image_file(p)]
    return paths