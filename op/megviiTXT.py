from __future__ import division

import os
import cv2
import json
import shutil
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def json2txt(annotation_file, path):
    dataset = json.load(open(annotation_file, 'r'))
    print dataset.keys()

    image_list = dataset['images']
    print 'num of images', len(image_list)

    ann_list = dataset['annotations']
    print 'num of annotations', len(ann_list)

    min = 366
    max = -1
    cate = dataset['categories']
    for i in range(len(cate)):
        if cate[i][u'id'] > max:
            max = cate[i][u'id']
        if cate[i][u'id'] < min:
            min = cate[i][u'id']
        print i, cate[i]

    print min, max

    names = list()
    imgs = list()
    maps = dict()
    idx = 0
    for itm in image_list:
        names.append(itm[u'file_name'].encode("utf-8"))
        imgs.append([itm[u'file_name'].encode("utf-8"), itm[u'height'], itm[u'width']])
        maps[itm[u'id']] = idx
        idx += 1

    for itm in ann_list:

        idx = maps[itm[u'image_id']]
        bbox = itm[u'bbox']

        cate_id = itm[u'category_id']

        if bbox[0] + bbox[2] >= imgs[idx][2]:
            bbox[2] = imgs[idx][2] - bbox[0] - 2  # make sure the bbox isn't outside of the image

        if bbox[1] + bbox[3] >= imgs[idx][1]:
            bbox[3] = imgs[idx][1] - bbox[1] - 2  # move 2 pixels

        bbox[0] += bbox[2] / 2.0
        bbox[1] += bbox[3] / 2.0

        bbox[0] = bbox[0] / float(imgs[idx][2])
        bbox[1] = bbox[1] / float(imgs[idx][1])
        bbox[2] = bbox[2] / float(imgs[idx][2])
        bbox[3] = bbox[3] / float(imgs[idx][1])

        assert cate_id - 1 >= 0
        assert cate_id - 1 < 65
        bbox.insert(0, cate_id - 1)  # such that index starts from 0
        imgs[idx].append(bbox)

    for i in range(len(imgs)):
        if len(imgs[i]) == 3:
            continue
        strm = open(os.path.join(path, '%s' % names[i].replace('jpg', 'txt')), 'w')
        for j in range(3, len(imgs[i])):
            strm.write('%s ' % imgs[i][j][0])
            for k in range(1, 5):
                strm.write('%.6f ' % imgs[i][j][k])
            strm.write('\n')
        strm.close()


def names(annotation_file, file):
    dataset = json.load(open(annotation_file, 'r'))
    cate = dataset['categories']
    maps = dict()

    for i in range(len(cate)):
        maps[cate[i][u'id']] = cate[i][u'name'].encode("utf-8")
    strm = open(file, 'w')
    for i in range(len(cate)):
        strm.write('%s\n' % maps[i+1])
    strm.close()

def check():
    fn = '/home/yeezy/Dat/Megvii/train.txt'

    boxes_list = list()
    box_cnt = 0
    with open(fn, 'r') as imgs:
        for img in imgs:
            src_img = img.split('\n')[0]
            src_label = src_img.replace('jpg', 'txt').replace('images', 'labels')
            img_fn = src_img.split('/')[-1]

            print src_img
            print src_label
            print img_fn

            image = cv2.imread(src_img, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[0:2]

            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(image)

            boxes = list()
            with open(src_label, 'r') as bbox:
                for box in bbox:
                    box = box.split(' ')[1:5]
                    box = [float(cord) for cord in box]
                    x1 = box[0] - box[2] / 2.0
                    x2 = box[0] + box[2] / 2.0
                    y1 = box[1] - box[3] / 2.0
                    y2 = box[1] + box[3] / 2.0
                    x1 *= width
                    x2 *= width
                    y1 *= height
                    y2 *= height
                    cord = [float(int(x1)), float(int(y1)), float(int(x2)), float(int(y2))]
                    boxes.append(cord)

                    bbox = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='blue',
                                             facecolor='none')
                    ax.add_patch(bbox)

            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig('/home/yeezy/Desktop/vis/{}'.format(img_fn), bbox_inches='tight', pad_inches=0.0)
            plt.close()

            box_cnt += len(boxes)
            boxes_list.append(boxes)


def copy(txt_path, src_path, dst_path):
    img_list = os.listdir(txt_path)
    for file in img_list:
        name = file.replace('txt', 'jpg')
        src_file = os.path.join(src_path, name)
        dst_file = os.path.join(dst_path, name)
        os.symlink(src_file, dst_file)


def txt(img_dir, txt_file):
    imgs = os.listdir(img_dir)
    imgs.sort()
    strm = open(txt_file, 'w')
    for img in imgs:
        strm.write('%s\n' % os.path.join(img_dir, img))

    strm.close()


def xtxt(label_dir, txt_file):
    imgs = os.listdir(label_dir)
    imgs.sort()
    strm = open(txt_file, 'w')
    for img in imgs:
        # strm.write('%s\n' % os.path.join(label_dir, img.replace('txt', 'jpg'))) # COCO
        fn = os.path.join(label_dir.replace('labels', 'images'), img.replace('txt', 'jpg'))
        strm.write('%s\n' % fn)  # Megvii

    strm.close()


if __name__ == "__main__":

    # # Tiny Megvii
    annotation_file = '/media/data1/lzhang/tiny_megvii/annotations/tiny_val.json'
    path = '/home/yz/cde/ProposalYOLO/data/tiny_megvii/labels/valid/'
    json2txt(annotation_file, path)

    annotation_file = '/media/data1/lzhang/tiny_megvii/annotations/tiny_train.json'
    path = '/home/yz/cde/ProposalYOLO/data/tiny_megvii/labels/train/'
    json2txt(annotation_file, path)

    txt_path = '/home/yz/cde/ProposalYOLO/data/tiny_megvii/labels/valid/'
    src_path = '/home/yz/cde/ProposalYOLO/data/megvii/images/val/'
    dst_path = '/home/yz/cde/ProposalYOLO/data/tiny_megvii/images/valid/'
    copy(txt_path, src_path, dst_path)

    txt_path = '/home/yz/cde/ProposalYOLO/data/tiny_megvii/labels/train/'
    src_path = '/home/yz/cde/ProposalYOLO/data/megvii/images/train/'
    dst_path = '/home/yz/cde/ProposalYOLO/data/tiny_megvii/images/train/'
    copy(txt_path, src_path, dst_path)

    annotation_file = '/media/data1/lzhang/tiny_megvii/annotations/tiny_val.json'
    names(annotation_file, '/home/yz/cde/ProposalYOLO/data/tiny_megvii/tiny.names')

    label_dir = '/home/yz/cde/ProposalYOLO/data/tiny_megvii/labels/valid/'
    txt_file = '/home/yz/cde/ProposalYOLO/data/tiny_megvii/valid.txt'
    xtxt(label_dir, txt_file)

    label_dir = '/home/yz/cde/ProposalYOLO/data/tiny_megvii/labels/train/'
    txt_file = '/home/yz/cde/ProposalYOLO/data/tiny_megvii/train.txt'
    xtxt(label_dir, txt_file)

    # Visualization
    # check()

