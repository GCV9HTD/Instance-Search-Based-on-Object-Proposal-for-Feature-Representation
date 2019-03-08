from __future__ import division

import os
import sys
import importlib
import numpy as np

import torch
import torch.nn as nn

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))

from nets.proposal_loss import ProposalLoss
from nets.proposal_model import ProposalModel
from nets.proposal_attention import ProposalAttention
from common.coco_val_dataset import COCOvalDataset
from common.utils import non_max_suppression, bbox_iou


def eval(config):
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
    val_loader   = torch.utils.data.DataLoader(COCOvalDataset(config["val_path"], (config["img_w"], config["img_h"])),
                                               batch_size=16,  # set batch size by 1
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=False)

    """ VALIDATION """
    total = 0.0
    proposal = 0.0
    correct = 0.0
    net.eval()
    img_cnt = 0
    recall_cnt = 0.0
    for step, samples in enumerate(val_loader):
        images, labels = samples["image"], samples["label"]
        with torch.no_grad():
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(val_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output)

        # one image at a time !!!

        for label_i in range(labels.size(0)):

            total_avg = 0
            correct_avg = 0

            # calculate total
            targets = labels[label_i]
            for tx, ty, tw, th in targets:
                if tw > 0:
                    total += 1
                    total_avg += 1
                else:
                    continue

            # calculate proposal
            if batch_detections[label_i] is None:
                continue

            img_cnt += 1
            predictions = batch_detections[label_i]
            proposal += predictions.size(0)

            # calculate correct
            for tx, ty, tw, th in targets:
                x1, x2 = config["img_w"] * (tx - tw / 2.0), config["img_w"] * (tx + tw / 2.0)
                y1, y2 = config["img_h"] * (ty - th / 2.0), config["img_h"] * (ty + th / 2.0)
                box_gt = [x1, y1, x2, y2, 1.0]
                box_gt = torch.from_numpy(np.array(box_gt)).float().cuda()

                best_iou = 0.0
                for pred_i in range(predictions.size(0)):
                    iou = bbox_iou(predictions[pred_i].unsqueeze(0), box_gt.unsqueeze(0))
                    iou = iou.item()
                    best_iou = max(iou, best_iou)
                if best_iou >= 0.5:
                    correct += 1
                    correct_avg += 1
            recall_cnt += float(correct_avg / float(total_avg))
        if (step + 1) % 100 == 0:
            print 'Total: %d\tProposal: %d\tCorrect: %d\tPrecision: %.4f\tRecall: %.4f' % (total, proposal, correct, correct / (proposal + 1e-6), correct / (total + 1e-6))

    precision = correct / (proposal + 1e-6)
    recall = correct / (total + 1e-6)
    fscore = (2.0 * precision * recall) / (precision + recall + 1e-6)

    print("Precision: %.4f\tRecall: %.4f\tFscore: %.4f" % (precision, recall, fscore))
    print("Avg Recall: %.4f" % (recall_cnt / float(img_cnt + 1e-6)))


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

    eval(config)


if __name__ == "__main__":
    main()
