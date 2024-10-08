import math
import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.metrics import AverageMeter
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from src import config as cfg
from src.utils.visualize import Visualizer


class Evaluator:
    def __init__(self, dataset, model) -> None:
        self.model = model
        self.dataset = dataset
        self.loss_fn = nn.MSELoss()
    
    def eval(self):
        metrics = {
            'ms_ssim': AverageMeter(),
            'psnrb': AverageMeter()
        }
        ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(cfg['device'])
        
        self.model.eval()
        with torch.no_grad():
            for i, (X, y) in enumerate(tqdm(self.dataset)):
                X = X.to(cfg['device'])
                y = y.to(cfg['device'])
                outs = self.model(X)
                
                if i in cfg['debug']['debug_idxs']:
                    Visualizer._debug_output(X, outs, cfg['debug']['debug_ouput'], mode='valid', idx=i)

                loss = self.loss_fn(outs, y).item()
                loss = 20 * math.log10(1.0 / math.sqrt(loss))

                metrics['ms_ssim'].update(ms_ssim(y, outs).item())
                metrics['psnrb'].update(loss)
            print(f'==> MS_SSIM: {metrics["ms_ssim"].avg: .5f}')
            print(f'==> PSNRB: {metrics["psnrb"].avg: .5f}')
            
        return metrics