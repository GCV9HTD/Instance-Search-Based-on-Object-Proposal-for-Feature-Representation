import os
import sys
import cv2
import json
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import torch

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from common.utils import bbox_iou


def voc():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    gnd_dir = '/home/yz/cde/ProposalYOLO/data/voc/Labels'
    roi_dir = '/home/yz/cde/MxRCNN/roi/voc100'
    img_dir = '/home/yz/cde/ProposalYOLO/data/voc/JPEGImages'

    rois = os.listdir(roi_dir)
    rois.sort()

    gnds = os.listdir(gnd_dir)
    gnds.sort()

    assert len(rois) == len(gnds)

    total = 0.0
    proposal = 0.0
    correct = 0.0

    for i in range(len(rois)):

        # 1 Prediction
        pred_boxes = np.loadtxt(os.path.join(roi_dir, rois[i]))

        # 2 Ground-truth
        cords = np.loadtxt(os.path.join(gnd_dir, gnds[i]))

        try:
            cords = cords[:, 1:]
        except:
            cords = cords[1:]
            cords = cords.reshape(1, cords.shape[0])

        # 3 Height & Width
        img = os.path.join(img_dir, gnds[i].split('.')[0] + '.jpg')
        print img
        im = cv2.imread(img, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        height, width = im.shape[:2]

        gt_boxes = np.zeros(cords.shape)
        gt_boxes[:, 0] = (cords[:, 0] - cords[:, 2] / 2) * width
        gt_boxes[:, 1] = (cords[:, 1] - cords[:, 3] / 2) * height
        gt_boxes[:, 2] = (cords[:, 0] + cords[:, 2] / 2) * width
        gt_boxes[:, 3] = (cords[:, 1] + cords[:, 3] / 2) * height

        if i < 10:
            # Debug purpose
            plt.figure()
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(im)
            ax2.imshow(im)
            for idx in range(gt_boxes.shape[0]):
                bbox = patches.Rectangle((gt_boxes[idx][0], gt_boxes[idx][1]), gt_boxes[idx][2] - gt_boxes[idx][0],
                                         gt_boxes[idx][3] - gt_boxes[idx][1], linewidth=2, edgecolor='blue',
                                         facecolor='none')
                ax1.add_patch(bbox)
            for idx in range(pred_boxes.shape[0]):
                bbox = patches.Rectangle((pred_boxes[idx][0], pred_boxes[idx][1]),
                                         pred_boxes[idx][2] - pred_boxes[idx][0],
                                         pred_boxes[idx][3] - pred_boxes[idx][1], linewidth=2, edgecolor='red',
                                         facecolor='none')
                ax2.add_patch(bbox)
            ax1.axis('off')
            ax2.axis('off')
            # plt.gca().xaxis.set_major_locator(NullLocator())
            # plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig('/home/yz/cde/ProposalYOLO/eval/RPN/voc100/{}'.format(gnds[i].split('.')[0]), bbox_inches='tight',
                        pad_inches=0.0)
            plt.close()

        total += gt_boxes.shape[0]
        proposal += pred_boxes.shape[0]
        for j in range(gt_boxes.shape[0]):

            best_iou = 0.0
            for k in range(pred_boxes.shape[0]):
                # print gt_boxes[j], pred_boxes[k]
                gt = torch.from_numpy(gt_boxes[j]).float().cuda()
                pd = torch.from_numpy(pred_boxes[k]).float().cuda()
                iou = bbox_iou(pd.unsqueeze(0), gt.unsqueeze(0))
                iou = iou.item()
                best_iou = max(iou, best_iou)
            if best_iou >= 0.5:
                correct += 1

        print total, proposal, correct, correct / total

    precision = correct / proposal
    recall = correct / total
    fscore = (2.0 * precision * recall) / (precision + recall)
    print("Precision: %.4f\tRecall: %.4f\tFscore: %.4f" % (precision, recall, fscore))


def json2txt():
    # for coco
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    annotation_file = '/home/yz/cde/MxRCNN/data/coco/annotations/instances_minival2014.json'
    dataset = json.load(open(annotation_file, 'r'))

    print dataset.keys()

    image_list = dataset['images']
    print len(image_list)

    ann_list = dataset['annotations']
    print len(ann_list)

    imgs = list()
    maps = dict()
    idx = 0
    names = list()
    for itm in image_list:
        names.append(itm[u'file_name'].encode("utf-8"))
        imgs.append([itm[u'file_name'].encode("utf-8"), itm[u'height'], itm[u'width']])
        maps[itm[u'id']] = idx
        idx += 1

    i = 0
    for itm in ann_list:

        idx = maps[itm[u'image_id']]
        bbox = itm[u'bbox']
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        imgs[idx].append(bbox)


    dir = '/home/yz/cde/ProposalYOLO/data/coco/5kxLabels'
    for i in range(len(imgs)):
        strm = open(os.path.join(dir, '%s' % names[i].replace('jpg', 'txt')), 'w')
        for j in range(3, len(imgs[i])):
            for k in range(4):
                strm.write('%s ' % imgs[i][j][k])
            strm.write('\n')
        strm.close()


def coco():
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'

    gnd_dir = '/home/yz/cde/ProposalYOLO/data/coco/5kxLabels'
    roi_dir = '/home/yz/cde/MxRCNN/roi/coco10'
    img_dir = '/home/yz/cde/ProposalYOLO/data/coco/images/val2014'

    fn = '/home/yz/cde/ProposalYOLO/data/coco/5kx.txt'

    rois = os.listdir(roi_dir)
    rois.sort()

    gnds = os.listdir(gnd_dir)
    gnds.sort()

    assert len(rois) == len(gnds)

    total = 0.0
    proposal = 0.0
    correct = 0.0

    fn_strm = open(fn, 'r')
    for i in range(len(rois)):
        # 0 Name
        line = fn_strm.readline()
        name = line.split('\n')[0]


        # 1 Prediction
        pred_boxes = np.loadtxt(os.path.join(roi_dir, rois[i]))

        # 2 Ground-truth
        cords = np.loadtxt(os.path.join(gnd_dir, name.replace('jpg', 'txt')))

        try:
            cords = cords[:, 0:]
        except:
            cords = cords[0:]
            cords = cords.reshape(1, cords.shape[0])

        # 3 Height & Width
        img = os.path.join(img_dir, name)
        print img
        im = cv2.imread(img, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        height, width = im.shape[:2]

        gt_boxes = cords
        if gt_boxes.shape == (1, 0):
            continue

        if i < 10:
            # Debug purpose
            plt.figure()
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(im)
            ax2.imshow(im)
            for idx in range(gt_boxes.shape[0]):
                bbox = patches.Rectangle((gt_boxes[idx][0], gt_boxes[idx][1]), gt_boxes[idx][2] - gt_boxes[idx][0],
                                         gt_boxes[idx][3] - gt_boxes[idx][1], linewidth=2, edgecolor='blue',
                                         facecolor='none')
                ax1.add_patch(bbox)
            for idx in range(pred_boxes.shape[0]):
                bbox = patches.Rectangle((pred_boxes[idx][0], pred_boxes[idx][1]),
                                         pred_boxes[idx][2] - pred_boxes[idx][0],
                                         pred_boxes[idx][3] - pred_boxes[idx][1], linewidth=1, edgecolor='red',
                                         facecolor='none')
                ax2.add_patch(bbox)
            ax1.axis('off')
            ax2.axis('off')
            # plt.gca().xaxis.set_major_locator(NullLocator())
            # plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig('/home/yz/cde/ProposalYOLO/eval/RPN/test10/{}'.format(gnds[i].split('.')[0]), bbox_inches='tight',
                        pad_inches=0.0)
            plt.close()
        # continue
        total += gt_boxes.shape[0]
        proposal += pred_boxes.shape[0]
        for j in range(gt_boxes.shape[0]):

            best_iou = 0.0
            for k in range(pred_boxes.shape[0]):
                # print gt_boxes[j], pred_boxes[k]
                gt = torch.from_numpy(gt_boxes[j]).float().cuda()
                pd = torch.from_numpy(pred_boxes[k]).float().cuda()
                gt = gt.unsqueeze(0)
                pd = pd.unsqueeze(0)
                iou = bbox_iou(pd, gt)
                iou = iou.item()
                best_iou = max(iou, best_iou)
            if best_iou >= 0.5:
                correct += 1

        print total, proposal, correct, correct / total

    precision = correct / proposal
    recall = correct / total
    fscore = (2.0 * precision * recall) / (precision + recall)
    print("Precision: %.4f\tRecall: %.4f\tFscore: %.4f" % (precision, recall, fscore))


if __name__ == '__main__':
    # voc()
    # json2txt()
    coco()