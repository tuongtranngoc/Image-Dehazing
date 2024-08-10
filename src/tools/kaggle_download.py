import json
from kaggle.api.kaggle_api_extended import KaggleApi

dataset = 'chrisviviers/cityscapes-leftimg8bit-trainvaltest'
save_dir = 'dataset/cityscapes-leftimg8bit-trainvaltest'
api_token = {"username":"","key":""}

api = KaggleApi(api_client=api_token)
api.authenticate()
api.dataset_download_files(dataset, save_dir)