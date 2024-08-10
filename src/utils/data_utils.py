import cv2
import torch
import numpy as np
from typing import Tuple, Dict, List

from src import config as cfg


class DataUtils:
    
    @classmethod
    def to_device(cls, data):
        if isinstance(data, torch.Tensor):
            return data.to(cfg['device'])
        elif isinstance(data, Tuple) or isinstance(data, List):
            for i, d in enumerate(data):
                if isinstance(d, torch.Tensor):
                    data[i] = d.to(cfg['device'])
                else:
                    Exception(f"{d} in {data} is not a tensor type")
            return data
        elif isinstance(data, torch.nn.Module):
            return data.to(cfg['device'])
        else:
            Exception(f"{data} is not a/tuple/list of tensor type")
            
    @classmethod
    def denormalize(cls, image):
        image = image * 255.0
        image = np.clip(image, 0, 255.)
        return image
    
    @classmethod
    def image_to_numpy(cls, image):
        if isinstance(image, torch.Tensor):
            if image.dim() > 3:
                image = image.squeeze()
            image = image.detach().cpu().numpy()
            if image.ndim == 3:
                image = image.transpose((1, 2, 0))
                image = cls.denormalize(image)
                image = np.ascontiguousarray(image, np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image.ndim == 2:
                image = np.ascontiguousarray(image, np.uint8)
            return image
        
        elif isinstance(image, np.ndarray):
            image = cls.denormalize(image)
            image = np.ascontiguousarray(image, np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        
        else:
            raise Exception(f"{image} is a type of {type(image)}, not numpy/tensor type")