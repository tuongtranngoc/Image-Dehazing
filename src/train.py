import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import config as cfg
from src.evaluate import Evaluator
from src.models.UnetAttention import UNet
from src.utils.metrics import AverageMeter
from src.utils.tensorboard import Tensorboard
from src.data.dehaze_dataset import DeHazeDataset
from src.models.perceptual import vgg19_perceptual, normalize_batch
from src.utils.visualize import Visualizer


class Trainer:
    def __init__(self) -> None:
        self.st_epoch = 0
        self.best_psnr = 0.0
        self.best_ms_ssim = 0.0
        self.loss_fun = torch.nn.L1Loss()
        self.perceptual_loss = torch.nn.MSELoss()

        self.create_dataloader()
        self.model = UNet(dropout=0.0).to(device=cfg['device'])
        self.perceptual_model = vgg19_perceptual(cfg['device'])
        self.evaluator = Evaluator(self.valid_dataloader, self.model)
    
    def create_dataloader(self):
        self.train_dataset = DeHazeDataset(mode='train', aug=cfg['train']['augmentation'], split=cfg['split_image'])
        self.valid_dataset = DeHazeDataset(mode='valid', aug=cfg['valid']['augmentation'], split=cfg['split_image'])

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=cfg['train']['batch_size'],
                                           shuffle=True,
                                           num_workers=cfg['train']['num_workers'],
                                           pin_memory=cfg['train']['pin_memory'])

        self.valid_dataloader = DataLoader(dataset=self.valid_dataset,
                                           batch_size=cfg['valid']['batch_size'],
                                           shuffle=False,
                                           num_workers=cfg['valid']['num_workers'],
                                           pin_memory=cfg['valid']['pin_memory'])
        
    def train(self):
        metrics = {
            'train_pixel_loss': AverageMeter(),
            'train_total_loss': AverageMeter(),
            'train_perceptual_loss': AverageMeter(),
        }
        
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=float(cfg['lr']))
        
        for epoch in range(1, cfg['epochs']):
            self.model.train()
            for b, (X, y) in enumerate(self.train_dataloader):
                X = X.to(cfg['device'], dtype=torch.float32)
                y = y.to(cfg['device'], dtype=torch.float32)
                self.optimizer.zero_grad()
                outs = self.model(X)
                
                # Visualizer._debug_output(X, outs, cfg['debug']['debug_ouput'], 'train', cfg['debug']['debug_idxs'])
                
                loss = self.loss_fun(outs, y)
                perceptual_outs = self.perceptual_model(normalize_batch(outs))
                perceptual_y = self.perceptual_model(normalize_batch(y))
                
                perceptual_loss = self.perceptual_loss(perceptual_outs, perceptual_y)
                
                total_loss = loss * 10 + perceptual_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                metrics['train_pixel_loss'].update(loss.item())
                metrics['train_perceptual_loss'].update(perceptual_loss.item())
                metrics['train_total_loss'].update(total_loss.item())
                
                print(f'Epoch {epoch} - batch {b}/{len(self.train_dataloader)} - pixel_loss: {loss.item(): .5f} - perceptual_loss: {perceptual_loss.item(): .5f} - total_loss: {total_loss.item(): .5f}', end='\r')
                
                Tensorboard.add_scalars('train_total_loss', epoch, train_loss=metrics['train_total_loss'].avg)
                Tensorboard.add_scalars('train_pixel_loss', epoch, train_unet_loss=metrics['train_pixel_loss'].avg)
                Tensorboard.add_scalars('train_perceptual_loss', epoch, train_perceptual_loss=metrics['train_perceptual_loss'].avg)
        
            print(f"Epoch {epoch} - train_pixel_loss: {metrics['train_pixel_loss'].avg: .5f} - train_perceptual_loss: {metrics['train_perceptual_loss'].avg: .5f} - train_total_loss: {metrics['train_total_loss'].avg: .5f}")
            
            if epoch % cfg['train']['eval_step'] == 0:
                eval_metrics = self.evaluator.eval()
                Tensorboard.add_scalars('ms_ssim', epoch, ms_ssim=eval_metrics['ms_ssim'].avg)
        
                current_ms_ssim = eval_metrics['ms_ssim'].avg
                if current_ms_ssim > self.best_ms_ssim:
                    self.best_ms_ssim = current_ms_ssim
                    self.save_ckpt(cfg['debug']['weight'], self.best_ms_ssim, epoch)              
                
    def save_ckpt(self, save_path, best_acc, epoch):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ckpt_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": best_acc,
            "epoch": epoch,
        }
        print(f"Saving checkpoint to {save_path}")
        torch.save(ckpt_dict, save_path)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()