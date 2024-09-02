import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm

import torch

from src import config as cfg
from src.utils.data_utils import DataUtils
from src.models.UnetAttention import UNet
from src.utils.visualize import Visualizer
from src.data.dehaze_dataset import TransformDeHaze, RandomSpliting, GridCellSpliting


class Predictor:
    def __init__(self) -> None:
        self.model = UNet().to(cfg['device'])
        self.model.load_state_dict(torch.load(cfg['debug']['weight'], map_location=cfg['device'])['model'])
        self.transform = TransformDeHaze()

    def patches_predict(self, image_path):
        size = cfg['train']['image_size']
        org_image = cv2.imread(image_path)
        org_images = GridCellSpliting(size)(org_image)

        pad = 0
        H, W, C = org_image.shape
        hr = round(H/size)
        wr = round(W/size)
        mask = np.ones(org_image.shape, dtype=org_image.dtype) * 255.0

        for (i, j), org_img in org_images.items():
            image = org_img[..., ::-1]
            transformed_image = self.transform.transform(image=image).unsqueeze(0).to(cfg['device'])
            encode_image = self.model(transformed_image)
            encode_image = DataUtils.image_to_numpy(encode_image)
            decoded_image = Visualizer.decode_image(encode_image, org_img)

            mask[(j*(H//hr)) + (j+1) * pad:((j+1)*(H//hr)) + (j+1) * pad, (i*(W//wr)) + (i+1) * pad:((i+1)*(W//wr)) + (i+1) * pad] = decoded_image

        Visualizer.save_image(org_image, mask, os.path.basename(image_path), cfg['debug']['prediction'])

    def single_predict(self, image_path):
        org_image = cv2.imread(image_path)
        image = org_image[..., ::-1]
        transformed_image = self.transform.transform(image=image).unsqueeze(0)
        encode_image = self.model(transformed_image)
        encode_image = DataUtils.image_to_numpy(encode_image)
        decoded_image = Visualizer.decode_image(encode_image, org_image)
        Visualizer.save_image(org_image, decoded_image, os.path.basename(image_path), cfg['debug']['prediction'])
    
    
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='Path to input images folder')
    args = parser.parse_args()    
    return args

    
if __name__ == "__main__":
    predictor = Predictor()
    args = cli()
    for img_pth in tqdm(glob.glob(os.path.join(args.input_folder, "*"))):
        predictor.single_predict(img_pth)
    
    
    