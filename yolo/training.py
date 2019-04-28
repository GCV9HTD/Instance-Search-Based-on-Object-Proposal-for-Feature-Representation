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

from nets.yolo import YOLO
from nets.proposal_loss import YOLOLoss
from common.megvii_trn_dataset import MegviitrnDataset
from common.megvii_val_dataset import MegviivalDataset

# Logging
prefix = 'Tiny'

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
    config["trn_global_step"] = config.get("start_step", 0)
    config["val_global_step"] = config.get("start_step", 0)
    is_training = False if config.get("export_onnx") else True

    # Load and initialize network
    net = YOLO(config, is_training=is_training)
    net.train(is_training)

    # Optimizer and learning rate
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
        yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    eval_losses = []
    for i in range(3):
        eval_losses.append(
            YOLOLoss(config["yolo"]["anchors"][i], config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    # DataLoader
    train_dataset = MegviitrnDataset(config["train_path"], (config["img_w"], config["img_h"]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"],
                                             shuffle=True, num_workers=4, pin_memory=False)

    eval_dataset = MegviivalDataset(config["eval_path"], (config["img_w"], config["img_h"]))
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=config["batch_size"],
                                                  shuffle=False, num_workers=4, pin_memory=False)

    # Learning rate
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["lr"]["decay_step"], gamma=config["lr"]["decay_gamma"])

    # Start the training loop
    logger.info("Start training.")
    for epoch in range(config["epochs"]):

        net.train()
        optimizer.zero_grad()
        lr_scheduler.step()
        avg_loss = 0
        logger.info("\tTrain.")
        for step, samples in enumerate(train_dataloader):
            images, labels = samples["image"], samples["label"]
            start_time = time.time()
            config["trn_global_step"] += 1

            outputs = net(images)
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []
            for _ in range(len(losses_name)):
                losses.append([])
            for i in range(3):
                _loss_item = yolo_losses[i](outputs[i], labels)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]
            avg_loss = avg_loss + loss.item()
            loss = loss / (config["accumulation"] * 1.0)
            loss.backward()
            if (step + 1) % config["accumulation"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % 400 == 0:
                _loss = loss.item()
                duration = float(time.time() - start_time)
                example_per_second = config["batch_size"] / duration
                lr = optimizer.param_groups[0]['lr']
                logger.info(
                    "epoch [%.3d] iter = %d loss = %.6f avg loss = %.6f example/sec = %.3f lr = %.5f " %
                    (epoch, step + 1, _loss, avg_loss / (step + 1), example_per_second, lr)
                )
                config["tensorboard_writer"].add_scalar("train avg loss", avg_loss / (step + 1), config["trn_global_step"])

                for i, name in enumerate(losses_name):
                    value = _loss if i == 0 else losses[i]
                    config["tensorboard_writer"].add_scalar('train'+ name, value, config["trn_global_step"])

        logger.info("Training average loss = %.6f" % (avg_loss / len(train_dataloader)))

        _save_checkpoint(net.state_dict(), config, name="Ep%04d-model.pth" % epoch)

        net.eval()
        avg_loss = 0
        logger.info("\tEval.")
        for step, samples in enumerate(eval_dataloader):
            images, labels = samples["image"], samples["label"]
            start_time = time.time()
            config["val_global_step"] += 1
            outputs = net(images)
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []
            for _ in range(len(losses_name)):
                losses.append([])
            for i in range(3):
                _loss_item = eval_losses[i](outputs[i], labels)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]
            avg_loss = avg_loss + loss.item()

            if (step + 1) % 100 == 0:
                _loss = loss.item()
                duration = float(time.time() - start_time)
                example_per_second = config["batch_size"] / duration

                logger.info(
                    "epoch [%.3d] iter = %d loss = %.6f avg loss = %.6f example/sec = %.3f " %
                    (epoch, step + 1, _loss, avg_loss / (step + 1), example_per_second)
                )
                config["tensorboard_writer"].add_scalar("eval avg loss", avg_loss / (step + 1), config["val_global_step"])

                for i, name in enumerate(losses_name):
                    value = _loss if i == 0 else losses[i]
                    config["tensorboard_writer"].add_scalar('eval'+ name, value, config["val_global_step"])

        logger.info("Eval average loss = %.6f" % (avg_loss / len(eval_dataloader)))

    logger.info("Bye~")


def main():
    """

    :return:
    """
    if len(sys.argv) != 2:
        logger.error("Usage: python training.py params.py")
        sys.exit()

    params_path = sys.argv[1]
    if not os.path.isfile(params_path):
        logger.error("no params file found! path: {}".format(params_path))
        sys.exit()

    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS

    config["batch_size"] *= (len(config["parallels"]))

    # Create sub_working_dir
    sub_working_dir = '{}/{}/size{}x{}_try{}/{}'.format(config['working_dir'], 'megvii',
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

    logger.info(config)
    train(config)


if __name__ == "__main__":
    main()
