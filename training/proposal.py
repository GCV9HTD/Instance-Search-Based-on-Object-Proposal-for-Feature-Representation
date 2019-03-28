from __future__ import division

import os
import sys
import time
import logging
import importlib
import numpy as np
from tensorboardX import SummaryWriter


import torch
import torch.nn as nn
import torch.optim as optim


MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))


from nets.proposal_loss import ProposalLoss
from nets.proposal_model import ProposalModel
from nets.proposal_attention import ProposalAttention
from common.coco_trn_dataset import COCOtrnDataset
from common.coco_val_dataset import COCOvalDataset
from common.utils import non_max_suppression, bbox_iou


prefix = 'ATTN'

logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

if len(logger.handlers) > 0:
    logger.handlers = []

fh = logging.FileHandler('../log/' + prefix + '.log')
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("[%(asctime)s %(filename)s] %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)


def _save_checkpoint(state_dict, config, name="model.pth"):

    checkpoint_path = os.path.join(config["sub_working_dir"], name)
    torch.save(state_dict, checkpoint_path)
    logger.info("Model checkpoint saved to %s" % checkpoint_path)


def _get_optimizer(config, net):
    """

    :param config:
    :param net:
    :return:
    """
    # Assign different lr for each layer
    base_params = list(map(id, net.backbone.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, net.parameters())

    if not config["lr"]["freeze_backbone"]:
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
            {"params": net.backbone.parameters(), "lr": config["lr"]["backbone_lr"]},
        ]
    else:
        logger.info("Freeze backbone's parameters.")
        for p in net.backbone.parameters():
            p.requires_grad = False
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
        ]

    # Initialize optimizer class
    if config["optimizer"]["type"] == "adam":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"])

    elif config["optimizer"]["type"] == "amsgrad":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"], amsgrad=True)

    elif config["optimizer"]["type"] == "rmsprop":
        optimizer = optim.RMSprop(params, weight_decay=config["optimizer"]["weight_decay"])

    else:
        # Default to sgd
        logger.info("Using SGD optimizer.")
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=config["optimizer"]["weight_decay"],
                              nesterov=(config["optimizer"]["type"] == "nesterov"))

    return optimizer


def train(config):
    """

    :param config:
    :return:
    """
    config["global_step"] = config.get("start_step", 0)
    is_training = False if config.get("export_onnx") else True

    # Load and initialize network
    # net = ProposalModel(config, is_training=is_training)
    net = ProposalAttention(config, is_training=is_training)
    net.train(is_training)

    # Optimizer
    optimizer = _get_optimizer(config, net)

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logger.info("Load pretrained weights from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(
            ProposalLoss(config["yolo"]["anchors"][i], (config["img_w"], config["img_h"]))
        )

    val_losses = []
    for i in range(3):
        val_losses.append(
            ProposalLoss(config["yolo"]["anchors"][i], (config["img_w"], config["img_h"]))
        )

    # DataLoader
    coco_train_dataset = COCOtrnDataset(config["train_path"], (config["img_w"], config["img_h"]))
    coco_train_loader = torch.utils.data.DataLoader(coco_train_dataset, batch_size=config["batch_size"],
                                                    shuffle=True, num_workers=4, pin_memory=True)

    coco_val_dataset = COCOvalDataset(config["val_path"], (config["img_w"], config["img_h"]))
    coco_val_loader = torch.utils.data.DataLoader(coco_val_dataset, batch_size=8,
                                                  shuffle=False, num_workers=4, pin_memory=True)

    # Learning rate
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["lr"]["decay_step"], gamma=config["lr"]["decay_gamma"])
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=(0.1 * config["lr"]["other_lr"]))

    # Start the training loop
    logger.info("Start training.")
    for epoch in range(config["epochs"]):

        """ TRAIN """
        net.train()
        # ---------------------- #
        # Method 2
        # optimizer.zero_grad()
        # ---------------------- #
        avg_loss = 0
        # lr_scheduler.step()
        for step, samples in enumerate(coco_train_loader):
            images, labels = samples["image"], samples["label"]
            start_time = time.time()
            config["global_step"] += 1

            # ---------------------- #
            # Method 1
            optimizer.zero_grad()
            # ---------------------- #
            outputs = net(images)
            losses_name = ["total_loss", "x", "y", "w", "h", "conf"]
            losses = []
            for _ in range(len(losses_name)):
                losses.append([])
            for i in range(3):
                _loss_item = yolo_losses[i](outputs[i], labels)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]  # only need total loss

            # ---------------------- #
            # Method 1
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            # Method 2
            # loss = loss / (config["accumulation"] * 1.0)
            # avg_loss = avg_loss + loss.item()
            # loss.backward()
            # if (step + 1) % config["accumulation"] == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            # ---------------------- #

            if step > 0 and step % 400 == 0:  # 400 for mini-batch:64
                _loss = loss.item()
                duration = float(time.time() - start_time)
                example_per_second = config["batch_size"] / duration
                lr = optimizer.param_groups[0]['lr']
                logger.info(
                    "epoch [%.3d] iter = %d loss = %.6f avg loss = %.6f example/sec = %.2f lr = %.6f "%
                    (epoch, step, _loss, avg_loss / (step + 1), example_per_second, lr)
                )
                config["tensorboard_writer"].add_scalar("lr", lr, config["global_step"])
                config["tensorboard_writer"].add_scalar("example/sec", example_per_second, config["global_step"])
                config["tensorboard_writer"].add_scalar("avg loss", avg_loss / (step + 1), config["global_step"])

                for i, name in enumerate(losses_name):
                    value = _loss if i == 0 else losses[i]
                    config["tensorboard_writer"].add_scalar(name, value, config["global_step"])
            lr_scheduler.step()
        logger.info("Training average loss = %.6f" % (avg_loss / len(coco_train_loader)))

        _save_checkpoint(net.state_dict(), config, name="Ep%04d-model.pth" % epoch)

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=(0.1 * config["lr"]["other_lr"]))

        """ VALIDATION """
        # if epoch % 2 == 0:
        #     continue

        logger.info("Validating ... ")
        total = 0.0
        proposal = 0.0
        correct = 0.0
        net.eval()
        for step, samples in enumerate(coco_val_loader):
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
                # calculate total
                targets = labels[label_i]
                for tx, ty, tw, th in targets:
                    if tw > 0:
                        total += 1
                    else:
                        continue

                # calculate proposal
                if batch_detections[label_i] is None:
                    continue
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

            if (step + 1) % 200 == 0:
                print 'Total: %d\tProposal: %d\tCorrect: %d\tPrecision: %.4f\tRecall: %.4f' % (total, proposal, correct, correct / (proposal + 1e-6), correct / (total + 1e-6))

        precision = correct / (proposal + 1e-6)
        recall = correct / (total + 1e-6)
        fscore = (2.0*precision*recall) / (precision + recall + 1e-6)

        logger.info("Precision: %.4f\tRecall: %.4f\tFscore: %.4f" % (precision, recall, fscore))

    logger.info("Bye~")


def main():
    """

    :return:
    """
    """ Args """
    if len(sys.argv) != 2:
        logger.error("Usage: python proposal.py params.py")
        sys.exit()

    params_path = sys.argv[1]
    if not os.path.isfile(params_path):
        logger.error("no params file found! path: {}".format(params_path))
        sys.exit()

    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS

    # Set mini-batch size
    config["batch_size"] *= len(config["parallels"])

    # Create sub_working_dir
    sub_working_dir = '{}/{}/size{}x{}_try{}/{}'.format(
        config['working_dir'], config['model_params']['backbone_name'], 
        config['img_w'], config['img_h'], config['try'],
        time.strftime("%Y%m%d%H%M%S", time.localtime()))
    if not os.path.exists(sub_working_dir):
        os.makedirs(sub_working_dir)
    config["sub_working_dir"] = sub_working_dir
    logger.info("sub working dir: %s" % sub_working_dir)

    # Creat tf_summary writer
    config["tensorboard_writer"] = SummaryWriter(sub_working_dir)
    logger.info("Please using 'python -m tensorboard.main --logdir={}'".format(sub_working_dir))

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))

    # Kick off training
    logger.info(config)
    train(config)


if __name__ == "__main__":
    main()

