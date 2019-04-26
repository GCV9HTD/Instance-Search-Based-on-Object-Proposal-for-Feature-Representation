import os
import cv2
import shutil
import scipy.io

import numpy as np
from numpy.core.records import fromarrays

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


if __name__ == '__main__':

    fn = '/home/yeezy/Dat/COCO/coco/5k.txt'
    label_dir = '/home/yeezy/Dat/COCO/coco/labels/val2014/'

    boxes_list = list()
    img_fn_list = list()
    box_cnt = 0
    with open(fn, 'r') as imgs:
        for img in imgs:
            src_img = img.split('\n')[0]
            dst_img = src_img.replace('val2014', 'minival2014')
            img_fn = dst_img.split('/')[-1].split('.')[0]

            src_label = os.path.join(label_dir, src_img.split('/')[-1].replace('jpg', 'txt'))
            dst_label = src_label.replace('val2014', 'minival2014')
            try:
                shutil.copyfile(src_label, dst_label)
                pass
            except:
                print 'no such file:', src_label
                continue
            shutil.copyfile(src_img, dst_img)

            img_fn_list.append(img_fn)

            image = cv2.imread(dst_img, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[0:2]

            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(image)

            boxes = list()
            with open(dst_label, 'r') as bbox:
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

                    bbox = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='blue', facecolor='none')
                    ax.add_patch(bbox)

            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig('/home/yeezy/Desktop/vis/{}'.format(img_fn), bbox_inches='tight', pad_inches=0.0)
            plt.close()

            box_cnt += len(boxes)
            boxes_list.append(boxes)

    # Save as .mat file
    matfn = '/home/yeezy/Src/matlab/OP/evaluation-metrics/data/coco_gt_data.mat'
    record = fromarrays([img_fn_list, boxes_list], names=['im', 'boxes'])
    scipy.io.savemat(matfn, {'num_annotations': np.array([float(box_cnt)]), 'impos': record})



