import os
import sys

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

import numpy as np
import torch
from torchvision import datasets, transforms

from common.coco_val_dataset import COCOvalDataset

dataset = COCOvalDataset('/home/yz/cde/ProposalYOLO/data/TinyTrack/train.txt', (416, 416), is_debug=False)

loader = torch.utils.data.DataLoader(dataset, batch_size=2048, num_workers=8, shuffle=False)

pop_mean = []
pop_std = []
for i, sample in enumerate(loader):
    # shape (batch_size, 3, height, width)
    img = sample['image']
    numpy_image = img.numpy()

    # shape (3,)
    batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
    batch_std0 = np.std(numpy_image, axis=(0, 2, 3))

    pop_mean.append(batch_mean)
    pop_std.append(batch_std0)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
pop_mean = np.array(pop_mean).mean(axis=0)
pop_std = np.array(pop_std).mean(axis=0)

print(pop_mean, pop_std)

# mean=[0.46017307, 0.42988664, 0.3871622], std=[0.27410635, 0.26597932, 0.27650332]
