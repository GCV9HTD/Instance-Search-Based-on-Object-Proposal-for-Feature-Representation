import os
import sys
import cv2
import random
import importlib
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 256)]

import torch
import torch.nn as nn

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))

from nets.proposal_loss import ProposalLoss
from nets.proposal_model import ProposalModel
from nets.proposal_attention import ProposalAttention
from common.coco_val_dataset import COCOvalDataset
from common.utils import non_max_suppression


def vis(config):
    """

    :param config:
    :return:
    """
    is_training = False

    # Load and initialize network
    # net = ProposalModel(config, is_training=is_training)
    net = ProposalAttention(config, is_training=is_training)

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)

        # YOLO loss with 3 scales
        val_losses = []
        for i in range(3):
            val_losses.append(
                ProposalLoss(config["yolo"]["anchors"][i], (config["img_w"], config["img_h"]))
            )

        # DataLoader
        val_loader = torch.utils.data.DataLoader(COCOvalDataset(config["val_path"], (config["img_w"], config["img_h"])),
                                                 batch_size=4, shuffle=False, num_workers=2, pin_memory=False)

        sample = None
        step_d = 1
        step_i = 1
        cnt = 3
        for i, sample in enumerate(val_loader):
            if i == cnt:
                break
            # Detection
            images, labels = sample["image"], sample["label"]
            with torch.no_grad():
                outputs = net(images)
                output_list = []
                for i in range(3):
                    output_list.append(val_losses[i](outputs[i]))
                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output)

            for idx, detections in enumerate(batch_detections):

                plt.figure()
                fig, ax = plt.subplots(1)
                ax.imshow(np.transpose(images[idx].numpy(), (1, 2, 0)))

                if detections is not None:

                    for x1, y1, x2, y2, conf in detections:
                        bbox_colors = random.sample(colors, len(batch_detections))
                        color = bbox_colors[idx]

                        pre_h, pre_w = config["img_h"], config["img_w"]

                        if x1 < 0:
                            x1 = 0
                        if x2 > pre_w:
                            x2 = pre_w
                        if y1 < 0:
                            y1 = 0
                        if y2 > pre_h:
                            y2 = pre_h

                        box_h = (y2 - y1)
                        box_w = (x2 - x1)

                        # Create a Rectangle patch
                        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')

                        # Add the bbox to the plot
                        ax.add_patch(bbox)
                        # ax.text(x1, y1, '{:.2f}'.format(conf), bbox=dict(facecolor=color, alpha=0.9), fontsize=8, color='white')
                else:
                    print 'Nothing detected.'

                # Save generated image with detections
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
                plt.savefig('vis10/img{}_detect.jpg'.format(step_d), bbox_inches='tight', pad_inches=0.0)
                plt.close()
                step_d += 1

            # Ground-truth
            for i, (image, label) in enumerate(zip(sample['image'], sample['label'])):
                plt.figure()
                fig, ax = plt.subplots(1)
                ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
                for l in label:
                    if l.sum() == 0:
                        continue
                    x1 = int((l[0] - l[2] / 2) * config["img_w"])
                    y1 = int((l[1] - l[3] / 2) * config["img_h"])
                    x2 = int((l[0] + l[2] / 2) * config["img_w"])
                    y2 = int((l[1] + l[3] / 2) * config["img_h"])

                    box_h = (y2 - y1)
                    box_w = (x2 - x1)

                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor='blue', facecolor='none')
                    ax.add_patch(bbox)
                    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
                plt.savefig('vis10/img{}_input.jpg'.format(step_i), bbox_inches='tight', pad_inches=0.0)
                plt.close()
                step_i += 1



def main():
    """

    :return:
    """
    if len(sys.argv) != 2:
        print("Usage: python proposal.py params.py")
        sys.exit()

    params_path = sys.argv[1]
    if not os.path.isfile(params_path):
        print("no params file found! path: {}".format(params_path))
        sys.exit()

    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))

    vis(config)


if __name__ == '__main__':
    main()