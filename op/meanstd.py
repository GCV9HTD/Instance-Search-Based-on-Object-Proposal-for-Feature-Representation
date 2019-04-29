import cv2
import random
from tqdm import tqdm
import numpy as np


train_txt_path = '/home/yz/cde/ProposalYOLO/data/tiny_megvii/train.txt'

img_h, img_w = 416, 416
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

with open(train_txt_path, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)
    length = len(lines)

    for i in tqdm(range(length)):
        img_path = lines[i].rstrip().split()[0]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_h, img_w))

        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis=3)

imgs = imgs.astype(np.float32)/255.

for i in range(3):
    pixels = imgs[:,:,i,:].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

means.reverse() # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))