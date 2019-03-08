from __future__ import division

import os
import sys
import cv2
import glob
import pickle
import random
import logging
import importlib
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

import torch
import torch.nn as nn


MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))


from nets.proposal_loss import ProposalLoss
from nets.proposal_model import ProposalModel
from nets.proposal_attention import ProposalAttention
from common.utils import non_max_suppression
from layers.roi_pool.modules.roi_pool import RoIPooling


cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 256)]


def qry_Oxford_Paris(config):
    """

    :param config:
    :return:
    """

    # File name
    img_fn = '/home/yz/cde/ProposalYOLO/feature/qry/img-qry-oxford-baseline.txt'
    ftr_fn = '/home/yz/cde/ProposalYOLO/feature/qry/ftr-qry-oxford-baseline.txt'
    img_strm = open(img_fn, 'a')
    ftr_strm = open(ftr_fn, 'a')

    # Load and initialize network
    is_training = False
    net = ProposalModel(config, is_training=is_training)
    net.train(is_training)
    ROIdelegator = RoIPooling(pooled_height=1, pooled_width=1, spatial_scale=1.0 / 32)

    # Forward hook
    layer = net.targeted_layer()
    features = list()

    def hook_feature(module, input, output):
        features.append(output[0])
        features.append(output[1])
        features.append(output[2])

    layer.register_forward_hook(hook_feature)
    ftr_strm.write('%d\n' % (net.backbone.layers_out_filters[-1]))

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logging.info("load checkpoint from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)
    else:
        raise Exception("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(
            ProposalLoss(config["yolo"]["anchors"][i], (config["img_w"], config["img_h"]))
        )

    # Prepare images path
    images_name = os.listdir(config["ground_truth"])
    images_name.sort()
    images_path = [os.path.join(config["ground_truth"], name) for name in images_name]
    if len(images_path) == 0:
        raise Exception("no image found in {}".format(config["qry_path"]))

    # Start inference
    batch_size = config["batch_size"]
    for step in range(0, len(images_path), batch_size):
        # preprocess
        images = []
        features = []
        name = ""

        for path in images_path[step*batch_size: (step+1)*batch_size]:
            logging.info("processing: {}".format(path))

            with open(path, 'r') as in_strm:
                infos = in_strm.readline()
                infos = [ele for ele in infos.split(' ')]

                name = infos[0]
                bbox = [int(float(ele)) for ele in infos[1:]]

            image = cv2.imread(os.path.join(config["qry_path"], name + '.jpg'), cv2.IMREAD_COLOR)
            ori_h, ori_w = image.shape[:2]

            if image is None:
                logging.error("read path error: {}. skip it.".format(path))
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_origin = image  # keep for save result
            image = cv2.resize(image, (config["img_w"], config["img_h"]), interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)


            w_scale = float(ori_w) / config["img_w"]
            h_scale = float(ori_h) / config["img_h"]

            bbox[0] = int(bbox[0] / w_scale)
            bbox[2] = int(bbox[2] / w_scale)
            bbox[1] = int(bbox[1] / h_scale)
            bbox[3] = int(bbox[3] / h_scale)

            rois = list()
            xrois = list()
            roi = list()
            roi.append(0)
            roi.append(int(bbox[0]))
            roi.append(int(bbox[1]))
            roi.append(int(bbox[2]))
            roi.append(int(bbox[3]))
            rois.append(roi)
            xrois.append(roi)

            images.append(image)

        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()
        rois = torch.Tensor(rois).cuda()

        # inference
        with torch.no_grad():
            _ = net(images)

        feature_map = features[2]

        output = ROIdelegator(feature_map, rois)
        output = output.view(output.size(0), output.size(1) * output.size(2) * output.size(3))
        output = output.data.cpu().numpy()
        output = normalize(output, norm='l2', axis=1)
        # print output.shape

        # Output
        cnt = output.shape[0]
        assert cnt == rois.shape[0]
        dim = output.shape[1]
        ori_h, ori_w = images_origin.shape[:2]
        pre_h, pre_w = config["img_h"], config["img_w"]
        for i in xrange(cnt):
            x1 = torch.clamp(rois[i][1].data, min=0, max=pre_w) / pre_w * ori_w
            y1 = torch.clamp(rois[i][2].data, min=0, max=pre_h) / pre_h * ori_h
            x2 = torch.clamp(rois[i][3].data, min=0, max=pre_w) / pre_w * ori_w
            y2 = torch.clamp(rois[i][4].data, min=0, max=pre_h) / pre_h * ori_h
            img_strm.write('%s %d %d %d %d\n' % (name, x1, y1, x2, y2))
            for j in xrange(dim):
                if j < dim - 1:
                    ftr_strm.write('%f ' % output[i][j])
                else:
                    ftr_strm.write('%f\n' % output[i][j])


        """
        # write result images. Draw bounding boxes and labels of detections
        if not os.path.isdir("./qry/"):
            os.makedirs("./qry/")

        for idx, box in enumerate(xrois):

            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(images_origin)

            if box is not None:
                # Rescale coordinates to original dimensions
                pre_h, pre_w = config["img_h"], config["img_w"]

                bbox_colors = random.sample(colors, 80)
                color = bbox_colors[idx]

                x1, y1, x2, y2 = box[1], box[2], box[3], box[4]

                box_h = ((y2 - y1) / pre_h) * ori_h
                box_w = ((x2 - x1) / pre_w) * ori_w
                y1 = (y1 / pre_h) * ori_h
                x1 = (x1 / pre_w) * ori_w

                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')

                # Add the bbox to the plot
                ax.add_patch(bbox)

            else:
                print 'Nothing detected.'

            # Save generated image with detections
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig('qry/{}.jpg'.format(name), bbox_inches='tight', pad_inches=0.0)
            plt.close()
        """
        # exit(0)

    # logging.info("Save all results to ./output/")


def pca(config):
    """

    :param config:
    :return:
    """

    # Load and initialize network
    is_training = False
    # net = ProposalModel(config, is_training=is_training)
    net = ProposalAttention(config, is_training=is_training)
    net.train(is_training)

    ROIdelegator = RoIPooling(pooled_height=1, pooled_width=1, spatial_scale=1.0 / 32)

    # Forward hook
    layer = net.targeted_layer()
    features = list()

    def hook_feature(module, input, output):
        features.append(output[0])
        features.append(output[1])
        features.append(output[2])

    layer.register_forward_hook(hook_feature)

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logging.info("load checkpoint from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)
    else:
        raise Exception("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(
            ProposalLoss(config["yolo"]["anchors"][i], (config["img_w"], config["img_h"]))
        )

    # Prepare images path
    images_path = [y for x in os.walk(config["pca_path"]) for y in glob.glob(os.path.join(x[0], "*.jpg"))]
    images_path.sort()
    print 'Num. of images: ', len(images_path)

    if len(images_path) == 0:
        raise Exception("no image found in {}".format(config["ref_path"]))

    # Start inference
    batch_size = config["batch_size"]
    feats = list()
    for step in range(0, len(images_path), batch_size):
        # preprocess
        images = []
        images_origin = []
        features = []
        name = ""
        for path in images_path[step*batch_size: (step+1)*batch_size]:
            logging.info("processing: {}".format(path))
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            name = path.split('/')[-1][:-4]
            if image is None:
                logging.error("read path error: {}. skip it.".format(path))
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_origin.append(image)  # keep for save result
            image = cv2.resize(image, (config["img_w"], config["img_h"]), interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            images.append(image)
        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()
        # inference
        with torch.no_grad():
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)

            batch_detections = non_max_suppression(output)

        # ROI Pooling
        if batch_detections[0] is None:
            continue
        rois = batch_detections[0].data.cpu().numpy()[:, :-1]
        image_index = np.zeros((rois.shape[0], 1))
        rois =  np.rint(np.concatenate((image_index, rois), axis=1))

        rois = torch.Tensor(rois).cuda()
        feature_map = features[2]

        output = ROIdelegator(feature_map, rois)
        output = output.view(output.size(0), output.size(1) * output.size(2) * output.size(3))
        output = output.data.cpu().numpy()
        # output = normalize(output, norm='l2', axis=1)
        # print output.shape

        # Output
        feats.append(output)

    feats = np.concatenate(tuple(feats), axis=0)
    print 'Learning pca with dim:', feats.shape

    # PCA
    feats = normalize(feats, norm='l2', axis=1)
    pca = PCA(feats.shape[1], whiten=True)
    pca.fit(feats)
    pickle.dump(pca, open(config["mapping_fn"], 'wb'))


def qry(config):
    """

    :param config:
    :return:
    """

    # File name
    img_fn = '/home/yz/cde/ProposalYOLO/feature/qry/img-qry-attn.txt'
    ftr_fn = '/home/yz/cde/ProposalYOLO/feature/qry/ftr-qry-attn.txt'
    img_strm = open(img_fn, 'a')
    ftr_strm = open(ftr_fn, 'a')

    # Load and initialize network
    is_training = False
    # net = ProposalModel(config, is_training=is_training)
    net = ProposalAttention(config, is_training=is_training)
    net.train(is_training)

    ROIdelegator = RoIPooling(pooled_height=1, pooled_width=1, spatial_scale=1.0 / 32)

    # Forward hook
    layer = net.targeted_layer()
    features = list()

    def hook_feature(module, input, output):
        features.append(output[0])
        features.append(output[1])
        features.append(output[2])

    layer.register_forward_hook(hook_feature)
    ftr_strm.write('%d\n' % (net.backbone.layers_out_filters[-1]))

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logging.info("load checkpoint from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)
    else:
        raise Exception("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(
            ProposalLoss(config["yolo"]["anchors"][i], (config["img_w"], config["img_h"]))
        )

    # Prepare images path
    images_name = os.listdir(config["qry_path"])
    images_name.sort()
    images_path = [os.path.join(config["qry_path"], name) for name in images_name]
    if len(images_path) == 0:
        raise Exception("no image found in {}".format(config["qry_path"]))

    # Start inference
    if config["pca"]:
        pca = pickle.load(open(config["mapping_fn"], 'rb'))
    batch_size = config["batch_size"]
    for step in range(0, len(images_path), batch_size):
        # preprocess
        images = []
        features = []
        name = ""

        for path in images_path[step*batch_size: (step+1)*batch_size]:
            logging.info("processing: {}".format(path))

            image = cv2.imread(path, cv2.IMREAD_COLOR)
            ori_h, ori_w = image.shape[:2]

            name = path.split('/')[-1][:-4]
            # Instance
            base_name = name.split('_')[0]
            # Instre
            # base_name = name

            if image is None:
                logging.error("read path error: {}. skip it.".format(path))
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_origin = image  # keep for save result
            image = cv2.resize(image, (config["img_w"], config["img_h"]), interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)

            bbox_fn = os.path.join(config["ground_truth"], base_name + '.txt')
            w_scale = float(ori_w) / config["img_w"]
            h_scale = float(ori_h) / config["img_h"]

            with open(bbox_fn, 'r') as in_strm:
                infos = in_strm.readline()
                infos = [ele for ele in infos.split('\t')]

                assert infos[0] == name
                bbox = [int(ele) for ele in infos[1:]]
                # print bbox
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                bbox[0] = int(bbox[0] / w_scale)
                bbox[2] = int(bbox[2] / w_scale)
                bbox[1] = int(bbox[1] / h_scale)
                bbox[3] = int(bbox[3] / h_scale)

            rois = list()
            xrois = list()
            roi = list()
            roi.append(0)
            roi.append(int(bbox[0]))
            roi.append(int(bbox[1]))
            roi.append(int(bbox[2]))
            roi.append(int(bbox[3]))
            rois.append(roi)
            xrois.append(roi)

            images.append(image)

        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()
        rois = torch.Tensor(rois).cuda()

        # inference
        with torch.no_grad():
            _ = net(images)

        feature_map = features[2]

        output = ROIdelegator(feature_map, rois)
        output = output.view(output.size(0), output.size(1) * output.size(2) * output.size(3))
        output = output.data.cpu().numpy()

        # Output
        cnt = output.shape[0]
        assert cnt == rois.shape[0]
        dim = output.shape[1]

        if config["pca"]:
            output = pca.transform(output)
        output = normalize(output, norm='l2', axis=1)

        ori_h, ori_w = images_origin.shape[:2]
        pre_h, pre_w = config["img_h"], config["img_w"]
        for i in xrange(cnt):
            x1 = torch.clamp(rois[i][1].data, min=0, max=pre_w) / pre_w * ori_w
            y1 = torch.clamp(rois[i][2].data, min=0, max=pre_h) / pre_h * ori_h
            x2 = torch.clamp(rois[i][3].data, min=0, max=pre_w) / pre_w * ori_w
            y2 = torch.clamp(rois[i][4].data, min=0, max=pre_h) / pre_h * ori_h
            img_strm.write('%s %d %d %d %d\n' % (name, x1, y1, x2, y2))
            for j in xrange(dim):
                if j < dim - 1:
                    ftr_strm.write('%f ' % output[i][j])
                else:
                    ftr_strm.write('%f\n' % output[i][j])


        """
        # write result images. Draw bounding boxes and labels of detections
        if not os.path.isdir("./qry/"):
            os.makedirs("./qry/")

        for idx, box in enumerate(xrois):

            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(images_origin)

            if box is not None:
                # Rescale coordinates to original dimensions
                pre_h, pre_w = config["img_h"], config["img_w"]

                bbox_colors = random.sample(colors, 80)
                color = bbox_colors[idx]

                x1, y1, x2, y2 = box[1], box[2], box[3], box[4]

                box_h = ((y2 - y1) / pre_h) * ori_h
                box_w = ((x2 - x1) / pre_w) * ori_w
                y1 = (y1 / pre_h) * ori_h
                x1 = (x1 / pre_w) * ori_w

                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')

                # Add the bbox to the plot
                ax.add_patch(bbox)

            else:
                print 'Nothing detected.'

            # Save generated image with detections
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig('qry/{}.jpg'.format(name), bbox_inches='tight', pad_inches=0.0)
            plt.close()
        """
        # exit(0)

    # logging.info("Save all results to ./output/")


def ref(config):
    """

    :param config:
    :return:
    """

    # File name
    img_fn = '/home/yz/cde/ProposalYOLO/feature/ref/img-ref-attn.txt'
    ftr_fn = '/home/yz/cde/ProposalYOLO/feature/ref/ftr-ref-attn.txt'
    img_strm = open(img_fn, 'a')
    ftr_strm = open(ftr_fn, 'a')

    # Load and initialize network
    is_training = False
    # net = ProposalModel(config, is_training=is_training)
    net = ProposalAttention(config, is_training=is_training)
    net.train(is_training)

    ROIdelegator = RoIPooling(pooled_height=1, pooled_width=1, spatial_scale=1.0 / 32)

    # Forward hook
    layer = net.targeted_layer()
    features = list()

    def hook_feature(module, input, output):
        features.append(output[0])
        features.append(output[1])
        features.append(output[2])

    layer.register_forward_hook(hook_feature)
    ftr_strm.write('%d\n' % (net.backbone.layers_out_filters[-1]))

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logging.info("load checkpoint from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)
    else:
        raise Exception("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(
            ProposalLoss(config["yolo"]["anchors"][i], (config["img_w"], config["img_h"]))
        )

    # Prepare images path
    images_path = [y for x in os.walk(config["ref_path"]) for y in glob.glob(os.path.join(x[0], "*.jpg"))]
    images_path.sort()
    print 'Num. of images: ', len(images_path)

    if len(images_path) == 0:
        raise Exception("no image found in {}".format(config["ref_path"]))

    # Start inference
    if config["pca"]:
        pca = pickle.load(open(config["mapping_fn"], 'rb'))
    batch_size = config["batch_size"]
    for step in range(0, len(images_path), batch_size):
        # preprocess
        images = []
        images_origin = []
        features = []
        name = ""
        for path in images_path[step*batch_size: (step+1)*batch_size]:
            logging.info("processing: {}".format(path))
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            name = path.split('/')[-1][:-4]
            if image is None:
                logging.error("read path error: {}. skip it.".format(path))
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_origin.append(image)  # keep for save result
            image = cv2.resize(image, (config["img_w"], config["img_h"]), interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            images.append(image)
        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()


        # inference
        with torch.no_grad():
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)

            batch_detections = non_max_suppression(output)

        # ROI Pooling
        if batch_detections[0] is None:
            continue
        rois = batch_detections[0].data.cpu().numpy()[:, :-1]
        image_index = np.zeros((rois.shape[0], 1))
        rois =  np.rint(np.concatenate((image_index, rois), axis=1))
        rois = torch.Tensor(rois).cuda()
        feature_map = features[2]

        output = ROIdelegator(feature_map, rois)
        output = output.view(output.size(0), output.size(1) * output.size(2) * output.size(3))
        output = output.data.cpu().numpy()

        # Output
        cnt = output.shape[0]
        assert cnt == rois.shape[0]
        dim = output.shape[1]

        if config["pca"]:
            output = pca.transform(output)
        output = normalize(output, norm='l2', axis=1)

        ori_h, ori_w = images_origin[0].shape[:2]
        pre_h, pre_w = config["img_h"], config["img_w"]
        for i in xrange(cnt):
            x1 = torch.clamp(rois[i][1].data, min=0, max=pre_w) / pre_w * ori_w
            y1 = torch.clamp(rois[i][2].data, min=0, max=pre_h) / pre_h * ori_h
            x2 = torch.clamp(rois[i][3].data, min=0, max=pre_w) / pre_w * ori_w
            y2 = torch.clamp(rois[i][4].data, min=0, max=pre_h) / pre_h * ori_h
            img_strm.write('%s-%04d %d %d %d %d\n' % (name, i + 1, x1, y1, x2, y2))
            for j in xrange(dim):
                if j < dim - 1:
                    ftr_strm.write('%f ' % output[i][j])
                else:
                    ftr_strm.write('%f\n' % output[i][j])


        """
        if step > 80:
            break
        # write result images. Draw bounding boxes and labels of detections
        if not os.path.isdir("./output/"):
            os.makedirs("./output/")

        for idx, detections in enumerate(batch_detections):
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(images_origin[idx])
            if detections is not None:
                # unique_labels = detections[:, -1].cpu().unique()
                # n_cls_preds = len(unique_labels)

                bbox_colors = random.sample(colors, detections.size(0))
                for x1, y1, x2, y2, conf in detections:
                    color = bbox_colors[idx]

                    # Rescale coordinates to original dimensions
                    ori_h, ori_w = images_origin[idx].shape[:2]
                    pre_h, pre_w = config["img_h"], config["img_w"]

                    if x1 < 0:
                        x1 = 0
                    elif x1 > pre_w:
                        x1 = pre_w

                    if x2 < 0:
                        x2 = 0
                    elif x2 > pre_w:
                        x2 = pre_w

                    if y1 < 0:
                        y1 = 0
                    elif y1 > pre_h:
                        y1 = pre_h

                    if y2 < 0:
                        y2 = 0
                    elif y2 > pre_h:
                        y2 = pre_h

                    box_h = ((y2 - y1) / pre_h) * ori_h
                    box_w = ((x2 - x1) / pre_w) * ori_w
                    y1 = (y1 / pre_h) * ori_h
                    x1 = (x1 / pre_w) * ori_w

                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')

                    # Add the bbox to the plot
                    ax.add_patch(bbox)

            else:
                print 'Nothing detected.'

            # Save generated image with detections
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig('output/{}.jpg'.format(name), bbox_inches='tight', pad_inches=0.0)
            plt.close()
        """

    # logging.info("Save all results to ./output/")


def main():
    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s %(filename)s] %(message)s")

    if len(sys.argv) != 2:
        logging.error("Usage: python images.py params.py")
        sys.exit()

    params_path = sys.argv[1]
    if not os.path.isfile(params_path):
        logging.error("no params file found! path: {}".format(params_path))
        sys.exit()

    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Start testing
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))

    """ PCA """
    # pca(config)

    """ Ref or Qry """
    # Instance-160, Instre
    qry(config)
    ref(config)

    """ Oxford & Paris """
    # qry_Oxford_Paris(config)
    # ref(config)


if __name__ == "__main__":
    main()
