import torch
import torchvision
import torch.nn as nn

from src import config as cfg


def vgg19_perceptual(device):
    CONTENT_LAYER = 'relu5_4'
    backbone = torchvision.models.vgg19(weights="DEFAULT").features
    model = nn.Sequential()
    
    i = 0
    for layer in backbone.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'            
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        
        model.add_module(name, layer)
        if name == CONTENT_LAYER:
            break
        
        model = model.to(device)
        model = torch.nn.DataParallel(model)

        for param in model.parameters():
            param.requires_grad = False

        for param in backbone.parameters():
            param.requires_grad = False
        
        return model
    

def normalize_batch(batch):
    # Normalize batch using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std
    