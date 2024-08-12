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
from src.data.derain_dataset import TransformDeReain, ImageSpliting


class Predictor:
    def __init__(self) -> None:
        self.model = UNet().to(cfg['device'])
        self.model.load_state_dict(torch.load(cfg['debug']['weight'], map_location=cfg['device'])['model'])
        self.transform = TransformDeReain()

    def patches_predict(self, image_path):
        size = cfg['train']['image_size']
        org_image = cv2.imread(image_path)
        org_images = ImageSpliting(size)(org_image)

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
            decoded_image = self.decode_image(encode_image, org_img)

            mask[(j*(H//hr)) + (j+1) * pad:((j+1)*(H//hr)) + (j+1) * pad, (i*(W//wr)) + (i+1) * pad:((i+1)*(W//wr)) + (i+1) * pad] = decoded_image

        self.save_image(org_image, mask, os.path.basename(image_path), cfg['debug']['prediction'])

    def single_predict(self, image_path):
        org_image = cv2.imread(image_path)
        image = org_image[..., ::-1]
        transformed_image = self.transform.transform(image=image).unsqueeze(0)
        encode_image = self.model(transformed_image)
        encode_image = DataUtils.image_to_numpy(encode_image)
        decoded_image = self.decode_image(encode_image, org_image)
        self.save_image(org_image, decoded_image, os.path.basename(image_path), cfg['debug']['prediction'])
        
    def decode_image(self, encode_image, org_image):
        size_img = cfg['train']['image_size']
        h, w = org_image.shape[:2]
        if max(h, w) > size_img:
                size_img = max(h, w) 
                encode_image = cv2.resize(encode_image, (size_img, size_img))
        
        h_pad = (size_img-h) // 2
        w_pad = (size_img-w) // 2
        out = encode_image[h_pad:h_pad+h, w_pad:w_pad+w]
        return out
    
    def save_image(self, org_image, decoded_image, basname, save_dir):
        pad = 20
        h, w, c = org_image.shape
        mask = np.ones(shape=(h, w*2 + pad, c), dtype=org_image.dtype) * 255
        
        mask[0:h, 0:w] = org_image
        mask[0:h, (w+pad): (w+pad)*2] = decoded_image
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, basname)
        cv2.imwrite(save_path, mask)
        

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='Path to input images folder')
    args = parser.parse_args()
    
    return args

    
if __name__ == "__main__":
    predictor = Predictor()
    args = cli()
    for img_pth in tqdm(glob.glob(os.path.join(args.input_folder, "*"))):
        predictor.patches_predict(img_pth)
    
    
    