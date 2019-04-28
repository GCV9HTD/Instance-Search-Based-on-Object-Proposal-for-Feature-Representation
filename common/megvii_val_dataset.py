import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from val_data_transform import *


class MegviivalDataset(Dataset):
    def __init__(self, list_path, img_size, is_debug=False):
        """

        :param list_path:
        :param img_size:
        :param is_debug:
        """
        self.img_files = []
        self.label_files = []

        for path in open(list_path, 'r'):
            label_path = path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').strip()
            if os.path.isfile(label_path):
                self.img_files.append(path)
                self.label_files.append(label_path)
            else:
                pass
                # print("no label found. skip it: {}".format(path))
        # print("Total images: {}".format(len(self.img_files)))
        self.img_size = img_size  # (w, h)
        self.max_objects = 50
        self.is_debug = is_debug

        #  transforms and augmentation
        self.transforms = Compose()
        # self.transforms.add(data_transforms.KeepAspect())
        self.transforms.add(ResizeImage(self.img_size))
        self.transforms.add(ToTensorX(self.max_objects, self.is_debug))

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Read image error: {}".format(img_path))
        ori_h, ori_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)  # reserve the class label
        else:
            # print("label does not exist: {}".format(label_path))
            labels = np.zeros((1, 5), np.float32)

        sample = {'image': img, 'label': labels}
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample["image_path"] = img_path
        sample["origin_size"] = str([ori_w, ori_h])
        return sample

    def __len__(self):
        return len(self.img_files)


if __name__ == "__main__":

    dataloader = torch.utils.data.DataLoader(MegviivalDataset("../data/tiny_megvii/val.txt", (416, 416), is_debug=True),
                                             batch_size=16, shuffle=True, num_workers=1, pin_memory=False)

    # dataloader = torch.utils.data.DataLoader(MegviiDataset("../data/coco/5k.txt", (416, 416), is_debug=True),
    #                                          batch_size=16, shuffle=False, num_workers=1, pin_memory=False)

    # print len(dataloader)
    for step, sample in enumerate(dataloader):
        print sample['label'].shape
        for i, (image, label) in enumerate(zip(sample['image'], sample['label'])):
            image = image.numpy()
            h, w = image.shape[:2]
            print 'class'
            for l in label:
                if l.sum() == 0:
                    continue
                print 'class id:', l[0].item()
                x1 = int((l[1] - l[3] / 2) * w)
                y1 = int((l[2] - l[4] / 2) * h)
                x2 = int((l[1] + l[3] / 2) * w)
                y2 = int((l[2] + l[4] / 2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
            print '\n'
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite("./demo/step{}_{}.jpg".format(step, i), image)
        # only one batch
        break
