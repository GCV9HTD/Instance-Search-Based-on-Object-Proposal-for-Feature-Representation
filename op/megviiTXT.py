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

        bbox[0] += bbox[2] / 2
        bbox[1] += bbox[3] / 2

        bbox[0] = bbox[0] / imgs[idx][2]
        bbox[1] = bbox[1] / imgs[idx][1]
        bbox[2] = bbox[2] / imgs[idx][2]
        bbox[3] = bbox[3] / imgs[idx][1]

        bbox.insert(0, cate_id)
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
                    x1 = box[0] - box[2] / 2
                    x2 = box[0] + box[2] / 2
                    y1 = box[1] - box[3] / 2
                    y2 = box[1] + box[3] / 2
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


if __name__ == "__main__":

    # annotation_file = '/home/yz/cde/ProposalYOLO/data/megvii/val.json'
    # path = '/home/yz/cde/ProposalYOLO/data/megvii/labels/val/'
    # json2txt(annotation_file, path)

    annotation_file = '/home/yz/cde/ProposalYOLO/data/megvii/train.json'
    path = '/home/yz/cde/ProposalYOLO/data/megvii/labels/train/'
    json2txt(annotation_file, path)

    # check()


