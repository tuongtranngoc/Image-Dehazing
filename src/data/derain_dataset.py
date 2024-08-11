
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
from src.utils.data_utils import DataUtils


class DeRainDataset(Dataset):
    def __init__(self, mode='train', aug=False, split='normal') -> None:
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

        if self.split == 'grid_cell':
            clean_imgs = GridCellSpliting(cfg['train']['image_size'])(clean_img)
            idxs = list(clean_imgs)
            idx = random.choice(idxs)
            clean_img = clean_imgs[idx]
            noise_img = GridCellSpliting()(noise_img)[idx]

        elif self.split == 'random_crop':
            H, W, C = clean_img.shape
            hight_left = H - cfg['train']['image_size']
            width_left = W - cfg['train']['image_size']

            r, c = 0, 0

            if hight_left > 0:
                r = np.random.randint(0, hight_left)
            if width_left > 0:
                c = np.random.randint(0, width_left)

            clean_img = RandomSpliting(cfg['train']['image_size'])(clean_img, r, c)
            noise_img = RandomSpliting(cfg['train']['image_size'])(noise_img, r, c)
        
        else:
            pass

        if self.aug:
            clean_img = self.transform.augment(clean_img)
            noise_img = self.transform.augment(noise_img)

        clean_img = self.transform.transform(clean_img)
        noise_img = self.transform.transform(noise_img)

        cv2.imwrite('img1.png', DataUtils.image_to_numpy(clean_img))
        cv2.imwrite('img2.png', DataUtils.image_to_numpy(noise_img))

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

        self._augment = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5)

        ])
    
    def transform(self, image):
        transformed = self._transform(image=image)
        transformed_img = transformed['image'] / 255.
        return transformed_img

    def augment(self, image):
        augmented = self._augment(image=image)
        augmented_img = augmented['image']
        return augmented_img
    

class GridCellSpliting:
    def __init__(self, size=512):
        self.size = size

    def __call__(self, image:np.ndarray):
        H, W, C = image.shape
        
        mosaics = defaultdict()

        hr = round(H/self.size)
        wr = round(W/self.size)

        grid_cells = itertools.product(range(0, wr), range(0, hr))

        for i, j in grid_cells:
            mosaics[(i, j)] = image[(j*(H//hr)):((j+1)*(H//hr)), (i*(W//wr)):((i+1)*(W//wr))]
        mosaics = dict(sorted(mosaics.items(), key = lambda x:x[0]))
        return mosaics
    

class RandomSpliting:
    def __init__(self, size=512):
        self.size = size

    def __call__(self, image:np.ndarray, r, c):
        image = image[r:r+self.size, c:c+self.size]

        return image
