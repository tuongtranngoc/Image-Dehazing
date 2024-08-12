from collections import defaultdict
import os
import cv2
import glob
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config as cfg
from src.data_generation.gen_imgaug import GenWithImgAug
from src.data_generation.gen_parking_lane import gen_lanes
from src.data_generation.gen_albumentation import GenWithAlbumentation
from src.data_generation.get_position_matrix import get_position_matrix


def composition_img(img,alpha,position_matrix,length=2):
    h, w = img.shape[0:2]
    dis_img = img.copy()

    for x in range(h):
        for y in range(w):
            u,v = int(position_matrix[0,x,y]/length),int(position_matrix[1,x,y]/length)
            if (u != 0 and v != 0):
                if((u<h) and (v<w)):
                    dis_img [x,y,:] = dis_img[u,v,:]
                elif(u<h):
                    print(w)
                    dis_img[x, y, :] = dis_img[u, np.random.randint(0,w-1), :]
                elif(v<w):
                    print(v)
                    dis_img[x, y, :] = dis_img[np.random.randint(0,h-1), v, :]

    dis_img = cv2.blur(dis_img,(3,3))*(0.9)

    img = (alpha/255)*dis_img + (1-(alpha/255))*img
    img = np.array(img,dtype=np.uint8)
    return img


def random_RainDrop(img, alpha_imgs, alpha_save_dir, texture_save_dir):
    alpha_img_name = alpha_imgs[random.randint(0,len(alpha_imgs)-1)]
    texture_img_name = 'texture_' + alpha_img_name
    
    alpha_img = cv2.imread(os.path.join(alpha_save_dir,alpha_img_name))
    texture_img = cv2.imread(os.path.join(texture_save_dir,texture_img_name))
    
    position_matrix, alpha = get_position_matrix(texture_img,alpha_img,img.shape[0:2],img)
    if np.random.uniform() > 0.5:
        img = GenWithAlbumentation._with_GlassBlur()(image=img)['image']
    img = composition_img(img,alpha,position_matrix)
    prefix = 'raindrop'
    return prefix, img


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha_dir', type=str, 
                        default='dataset/Honda-data/alpha_textures/alpha', 
                        help="Path to alpha directory")
    parser.add_argument('--texture_dir', type=str, 
                        default='dataset/Honda-data/alpha_textures/texture', 
                        help="Path to alpha directory")
    parser.add_argument('--mode', type=str,
                        default='valid',
                        help='dataset mode: train/valid/test')
    parser.add_argument('--original_dir', type=str, 
                        default='dataset/Honda-data/nuimages/dark_cam_back/original_data/', 
                        help='Path to original directory')
    parser.add_argument('--generated_dir', type=str, 
                        default=f'dataset/Honda-data/nuimages/dark_cam_back/{cfg["generated_version"]}', 
                        help='Path to output directory')
    args = parser.parse_args()
    
    
    return args
    

if __name__ == "__main__":
    args = cli()
    alpha_save_dir = args.alpha_dir
    texture_save_dir = args.texture_dir
    mode = args.mode
    original_dir = os.path.join(args.original_dir, mode)
    generated_dir = os.path.join(args.generated_dir, mode)
    
    os.makedirs(generated_dir, exist_ok=True)
    alpha_imgs = os.listdir(alpha_save_dir)
    
    statistic_dataset = defaultdict(lambda: defaultdict(int))
    
    for img_path in tqdm(glob.glob(os.path.join(original_dir, "*"))):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        
        if np.random.uniform() > 0.5:
            img = gen_lanes(img)

        aug = np.random.choice(['raindrop', 'albumentation', 'imgaug'])
        if aug == 'raindrop':
            prefix, img = random_RainDrop(img, alpha_imgs, alpha_save_dir, texture_save_dir)
        elif aug == 'albumentation':
            prefix, img = GenWithAlbumentation.compose_transformation(img)
        elif aug == 'imgaug':
            prefix, img = GenWithImgAug.compose_transformation(img)
            
        statistic_dataset[mode][prefix] += 1
                
        prefix_folder = os.path.join(generated_dir, prefix)
        os.makedirs(prefix_folder, exist_ok=True)
        cv2.imwrite(os.path.join(prefix_folder, f'{basename}.png'), img)
    os.makedirs(cfg['debug']['dataset_statistic'], exist_ok=True)    
    pd.DataFrame(statistic_dataset).to_csv(os.path.join(cfg['debug']['dataset_statistic'], os.path.basename(original_dir), f'{mode}.csv'))