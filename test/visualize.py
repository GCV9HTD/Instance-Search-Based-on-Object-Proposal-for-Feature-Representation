import os
import sys
import cv2
import importlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
np.set_printoptions(threshold='nan')

from nets.proposal_model import ProposalModel
from nets.proposal_attention import ProposalAttention


def vis(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))

    model = ProposalModel(config, is_training=False)
    # model = ProposalAttention(config, is_training=False)
    model.train(False)


    """ Backbone """
    layer = model.targeted_layer()
    features_bb = list()
    def hook_feature(module, input, output):
        # features_bb.append(output[2].data.cpu().squeeze(0))
        # features_bb.append(output[1].data.cpu().squeeze(0))
        features_bb.append(output[0].data.cpu().squeeze(0))
    layer.register_forward_hook(hook_feature)


    """ Embedding """
    attn_layer = model._modules.get('embedding2')[5]  # 2 -> Attention, 5 -> Baseline
    features_attn = list()
    def hook_feature(module, input, output):
        features_attn.append(output.data.cpu().squeeze(0))
    attn_layer.register_forward_hook(hook_feature)


    """ Set data parallel """
    model = nn.DataParallel(model)
    model = model.cuda()

    if config["pretrain_snapshot"]:
        state_dict = torch.load(config["pretrain_snapshot"])
        model.load_state_dict(state_dict)
    else:
        raise Exception("missing pretrain_snapshot!!!")


    """ Input """
    # prepare images path
    images_name = os.listdir(config["visualize_path"])
    images_name.sort()
    images_path = [os.path.join(config["visualize_path"], name) for name in images_name]
    print images_path
    if len(images_path) == 0:
        raise Exception("no image found in {}".format(config["images_path"]))

    # preprocess
    images = []
    images_origin = []
    path = images_path[0]

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images_origin.append(image)  # keep for save result
    image = cv2.resize(image, (config["img_w"], config["img_h"]), interpolation=cv2.INTER_LINEAR)

    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    plt.axis('off')
    plt.savefig('vis/input.jpg')
    plt.close()

    image = image.astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)

    images.append(image)
    images = np.asarray(images)
    images = torch.from_numpy(images).cuda()

    x = torch.randn(1, 3, 416, 416)
    y0, y1, y2 = model(images)


    """ Backbone & Embedding """
    c, h, w = features_attn[0].size()

    for i in range(c):
        im1 = features_bb[0].numpy()[i]
        im2 = features_attn[0].numpy()[i]
        plt.figure()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(im1)
        ax2.imshow(im2)
        ax1.axis('off')
        ax2.axis('off')
        plt.savefig('vis/base-channel%04d.jpg' % (i + 1))
        plt.close()


def main():

    config = importlib.import_module("params").TRAINING_PARAMS
    vis(config)
    

if __name__ == "__main__":
    main()
