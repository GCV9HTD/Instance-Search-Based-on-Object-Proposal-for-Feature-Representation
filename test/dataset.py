import os
import cv2
import glob
import numpy as np

import torch
import torch.utils.data
import torchvision.transforms as transforms

class Instance(torch.utils.data.Dataset):
    def __init__(self, img_dir, key='*.jpg', sz=(416, 416), gt_dir = None, transform=transforms.Compose([transforms.ToTensor(), ])):

        super(Instance, self).__init__()

        self.img_dir = img_dir
        self.key = key
        self.sz = sz
        self.gt_dir = gt_dir
        self.transform = transform

        self.list_img = [y for x in os.walk(self.img_dir) for y in glob.glob(os.path.join(x[0], key))]
        self.list_img.sort()

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        sample = dict()
        images = list()

        path = self.list_img[idx]
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        name = path.split('/')[-1][:-4].split('_')[0]
        ori_h, ori_w = image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.sz[0], self.sz[1]), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        images.append(image)

        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()

        if self.gt_dir:
            bbox_fn = os.path.join(self.gt_dir, name + '.txt')
            w_scale = float(ori_w) / float(self.sz[0])
            h_scale = float(ori_h) / float(self.sz[1])

            with open(bbox_fn, 'r') as in_strm:
                infos = in_strm.readline()
                infos = [ele for ele in infos.split('\t')]

                print infos[0], path.split('/')[-1][:-4]
                assert infos[0] == path.split('/')[-1][:-4]
                bbox = [int(ele) for ele in infos[1:]]
                # print bbox
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                bbox[0] = int(bbox[0] / w_scale)
                bbox[2] = int(bbox[2] / w_scale)
                bbox[1] = int(bbox[1] / h_scale)
                bbox[3] = int(bbox[3] / h_scale)
            sample["bbox"] = bbox

        sample["name"] = name
        sample["origin_size"] = (ori_w, ori_h)
        sample["images"] = images

        return sample

if __name__ == "__main__":

    img_dir = "/home/yz/cde/ProposalYOLO/data/Instance/qry"
    gt_dir = "/home/yz/cde/ProposalYOLO/data/Instance/bbox"

    dtst = Instance(img_dir=img_dir, gt_dir=gt_dir)

    loader = torch.utils.data.DataLoader(dtst, batch_size=1, shuffle=False, num_workers=1)

    for step, sample in enumerate(loader):
        print sample
        exit(0)




