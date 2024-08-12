import cv2
import os
import glob
import numpy as np

from src import config as cfg
from src.utils.data_utils import DataUtils


class Visualizer:
    image_size = cfg['train']['image_size']
    
    @classmethod
    def _debug_output(cls, inps, outs, debug_dir, mode, idx):
        os.makedirs(os.path.join(debug_dir, mode), exist_ok=True)
        
        for i, (inp, out) in enumerate(zip(inps, outs)):
            inp = DataUtils.image_to_numpy(inp)
            out = DataUtils.image_to_numpy(out)
            out = Visualizer.decode_image(inp, out)
            
            cls.save_image(inp, out, f'{idx}_{i}.png', os.path.join(debug_dir, mode))
            
    @classmethod
    def decode_image(cls, encode_image, org_image):
        size_img = cfg['train']['image_size']
        h, w = org_image.shape[:2]
        if max(h, w) > size_img:
                size_img = max(h, w) 
                encode_image = cv2.resize(encode_image, (size_img, size_img))
        
        h_pad = (size_img-h) // 2
        w_pad = (size_img-w) // 2
        out = encode_image[h_pad:h_pad+h, w_pad:w_pad+w]
        return out
    
    @classmethod
    def save_image(cls, org_image, decoded_image, basname, save_dir):
        pad = 20
        h, w, c = org_image.shape
        mask = np.ones(shape=(h, w*2 + pad, c), dtype=org_image.dtype) * 255
        
        mask[0:h, 0:w] = org_image
        mask[0:h, (w+pad): (w+pad)*2] = decoded_image
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, basname)
        cv2.imwrite(save_path, mask)