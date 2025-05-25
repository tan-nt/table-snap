import os
from ai_models.split_merge.dataset.dataset import ImageDataset
from ai_models.split_merge.modules.split_modules import SplitModel
import json
from PIL import Image
import torch
from torchsummary import summary
import numpy as np
import cv2
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# load dataset
folder = 'train'
with open('D:/dataset/table/table_line/Split1/'+ folder+'_labels.json', 'r') as f:
    labels = json.load(f)
dataset = ImageDataset('D:/dataset/table/table_line/Split1/'+ folder+'_input', labels, 8, scale=0.25)