import os 
import cv2
import glob
from tqdm import tqdm


DATA_PATH = "dataset/unimages/original_data/valid"
SAVE_DIR = "dataset/unimages/cvt_original_data/valid"
os.makedirs(SAVE_DIR, exist_ok=True)

for img_path in tqdm(glob.glob(os.path.join(DATA_PATH, "*"))):
    basename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    cv2.imwrite(os.path.join(SAVE_DIR, basename.replace('jpg', 'png')), img)
