import os
import cv2
import glob
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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


if __name__ == "__main__":

    # Change this to the folder of cityscape images.
    texture_save_dir = 'dataset/alpha_textures/texture'
    alpha_save_dir = 'dataset/alpha_textures/alpha'
    gen_raindrop = 'dataset/cityscapes-leftimg8bit-trainvaltest/generated_data/train'
    original_dir = 'dataset/cityscapes-leftimg8bit-trainvaltest/original_data/train'
    os.makedirs(gen_raindrop, exist_ok=True)
    alpha_imgs = os.listdir(alpha_save_dir)

    for img_path in tqdm(glob.glob(os.path.join(original_dir, "*"))):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if np.random.uniform() > 0.5:

            alpha_img_name = alpha_imgs[random.randint(0,len(alpha_imgs)-1)]
            texture_img_name = 'texture_' + alpha_img_name
            
            alpha_img = cv2.imread(os.path.join(alpha_save_dir,alpha_img_name))
            texture_img = cv2.imread(os.path.join(texture_save_dir,texture_img_name))
            
            position_matrix, alpha = get_position_matrix(texture_img,alpha_img,img.shape[0:2],img)
            if np.random.uniform() > 0.5:
                img = GenWithAlbumentation._with_GlassBlur()(image=img)['image']
            img = composition_img(img,alpha,position_matrix)
            prefix = 'raindrop'

        else:
            prefix, img = GenWithAlbumentation.compose_transformation(img)
        
        prefix_folder = os.path.join(gen_raindrop, prefix)
        os.makedirs(prefix_folder, exist_ok=True)
        cv2.imwrite(os.path.join(prefix_folder, f'{basename}.png'), img)