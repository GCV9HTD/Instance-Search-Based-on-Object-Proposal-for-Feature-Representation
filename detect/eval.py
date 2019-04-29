from __future__ import division

import os
import sys
import cv2
import json
import random
import importlib
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 80)]

import torch
import torch.nn as nn

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))

from nets.proposal_loss import YOLOLoss
from nets.yolo import YOLO
from common.megvii_val_dataset import MegviivalDataset
from common.utils import non_max_suppressionx, bbox_iou


def inference(config, val_json, res_json):
    """

    :param config:
    :return:
    """
    dataset = json.load(open(val_json, 'r'))
    image_list = dataset['images']
    id_map = dict()
    for itm in image_list:
        name = itm[u'file_name'].encode("utf-8")
        id = itm[u'id']
        id_map[name] = id

    is_training = False

    net = YOLO(config, is_training=is_training)

    net = nn.DataParallel(net)
    net = net.cuda()

    if config["pretrain_snapshot"]:
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)

    val_losses = []
    for i in range(3):
        val_losses.append(
            YOLOLoss(config["yolo"]["anchors"][i], config["yolo"]["classes"], (config["img_w"], config["img_h"]))
        )

    val_data = MegviivalDataset(config["eval_path"], (config["img_w"], config["img_h"]))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=False)

    classes = open(config["classes_names_path"], "r").read().split("\n")[:-1]

    json_list = list()

    for step, samples in enumerate(val_loader):
        print 'Step', step
        images, path, sz = samples["image"], samples["image_path"], samples["origin_size"]
        cate_id = path[0].split('/')[-1].split('.')[0]
        sz = [sz[0].item(), sz[1].item()] # height and width

        # print step, cate_id, sz

        with torch.no_grad():
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(val_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppressionx(output, config["yolo"]["classes"],
                                                    conf_thres=config["confidence_threshold"],
                                                    nms_thres=0.5)

        # print images.squeeze_(0).size()
        for idx, detections in enumerate(batch_detections):

            #----------------------------------------------------------------------------------------------------------
            plt.figure()
            fig, ax = plt.subplots(1)
            image = cv2.imread(path[0], cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            # ----------------------------------------------------------------------------------------------------------

            if detections is not None:

                # ----------------------------------------------------------------------------------------------------------
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                # ----------------------------------------------------------------------------------------------------------

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                    ori_h, ori_w = sz
                    pre_h, pre_w = config["img_h"], config["img_w"]
                    box_h = ((y2 - y1) / pre_h) * ori_h
                    box_w = ((x2 - x1) / pre_w) * ori_w
                    y1 = (y1 / pre_h) * ori_h
                    x1 = (x1 / pre_w) * ori_w
                    result_dict = dict()
                    result_dict['image_id'] = id_map[cate_id + '.jpg']
                    result_dict['category_id'] = int(cls_pred + 1)
                    result_dict['score'] = conf.item()
                    result_dict['bbox'] = [x1.item(), y1.item(), box_w.item(), box_h.item()]
                    json_list.append(result_dict)

                    # ----------------------------------------------------------------------------------------------------------
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(x1, y1, s=classes[int(cls_pred)] + ' %.2f' % (conf.item()),
                             color='white', verticalalignment='top', bbox={'color': color, 'pad': 0})
                    # ----------------------------------------------------------------------------------------------------------

            # ----------------------------------------------------------------------------------------------------------
            # Save generated image with detections
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig('output/{}.png'.format(cate_id), bbox_inches='tight', pad_inches=0.0)
            plt.close()
            # ----------------------------------------------------------------------------------------------------------

    with open(res_json, 'w') as f:
        json.dump(json_list, f)


def coco_eval(val_json, res_json):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    annType = 'bbox'
    annFile = val_json
    cocoGT = COCO(annFile)

    resFile = res_json
    cocoDT = cocoGT.loadRes(resFile)

    imgIds = sorted(cocoGT.getImgIds())
    imgIds = imgIds[0:100]

    cocoEval = COCOeval(cocoGT, cocoDT, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def main():
    """

    :return:
    """
    """ Args """
    if len(sys.argv) != 2:
        print("Usage: python proposal.py params.py")
        sys.exit()

    params_path = sys.argv[1]
    if not os.path.isfile(params_path):
        print("no params file found! path: {}".format(params_path))
        sys.exit()

    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))

    val_json = config["val_json"]
    res_json = config["res_json"]

    inference(config, val_json, res_json)

    coco_eval(val_json, res_json)


if __name__ == '__main__':

    main()
