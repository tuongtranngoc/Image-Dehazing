import math
import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.metrics import AverageMeter
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from src import config as cfg


class Evaluator:
    def __init__(self, dataset, model) -> None:
        self.model = model
        self.dataset = dataset
        self.loss_fn = nn.MSELoss()
    
    def eval(self):
        metrics = {
            'ms_ssim': AverageMeter(),
        }
        ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(cfg['device'])
        
        self.model.eval()
        with torch.no_grad():
            for (X, y) in tqdm(self.dataset):
                X = X.to(cfg['device'])
                y = y.to(cfg['device'])
                outs = self.model(X)
                metrics['ms_ssim'].update(ms_ssim(y, outs).item())
            print(f'==> MS_SSIM: {metrics["ms_ssim"].avg: .5f}')
            
        return metrics