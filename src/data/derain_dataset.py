
import albumentations as A
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch.transforms import ToTensorV2

import os
import cv2
import PIL
import glob
import random
import itertools
import numpy as np

from src import config as cfg


class DeRainDataset(Dataset):
    def __init__(self, mode='train', aug=False, split=True) -> None:
        self.split = split
        self.mode = mode
        self.aug = aug
        self.transform = TransformDeReain()
        self.dataset = self.get_dataset()
    
    def get_dataset(self):
        dataset = []
        for data_folder in cfg['data_dir']:
            clean_data = os.path.join(data_folder, 'original_data')
            noise_data = os.path.join(data_folder, 'generated_data')
            
            for img_path in glob.glob(os.path.join(noise_data, self.mode, "*", "*")):
                basename = os.path.basename(img_path)
                dataset.append({
                    'noise_path': img_path,
                    'clean_path': os.path.join(clean_data, self.mode, basename)
                })
        return dataset
                
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        clean_img = cv2.imread(self.dataset[index]['clean_path'])[..., ::-1]
        noise_img = cv2.imread(self.dataset[index]['noise_path'])[..., ::-1]

        if self.split:
            clean_imgs = ImageSpliting(cfg['train']['image_size'])(clean_img)
            idxs = list(clean_imgs)
            idx = random.choice(idxs)
            clean_img = clean_imgs[idx]
            noise_img = ImageSpliting()(noise_img)[idx]
        
        if self.aug:
            # Do something
            pass

        clean_img = self.transform.transform(clean_img)
        noise_img = self.transform.transform(noise_img)

        return noise_img, clean_img
        
        
class TransformDeReain:
    def __init__(self) -> None:
        """
        Reference: https://github.com/albumentations-team/albumentations/issues/718
        """
        self._transform = A.Compose([
            A.LongestMaxSize(max_size=cfg['train']['image_size'], interpolation=1),
            A.PadIfNeeded(min_height=cfg['train']['image_size'], min_width=cfg['train']['image_size'], border_mode=0, value=(0, 0, 0)),
            ToTensorV2()
        ])
    
    def transform(self, image):
        transformed = self._transform(image=image)
        transformed_img = transformed['image'] / 255.
        return transformed_img
    

class ImageSpliting:
    def __init__(self, size=512):
        self.size = size

    def __call__(self, image:np.ndarray):
        H, W, C = image.shape
        
        mosaics = defaultdict()

        hr = round(H/self.size)
        wr = round(W/self.size)

        gird_cells = itertools.product(range(0, wr), range(0, hr))

        for i, j in gird_cells:
            mosaics[(i, j)] = image[(j*(H//hr)):((j+1)*(H//hr)), (i*(W//wr)):((i+1)*(W//wr))]
        mosaics = dict(sorted(mosaics.items(), key = lambda x:x[0]))
        return mosaics