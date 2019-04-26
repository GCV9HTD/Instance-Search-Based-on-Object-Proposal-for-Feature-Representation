import os
import shutil
import random
import xml.sax


import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


class ImageNetHandler(xml.sax.ContentHandler):

    def __init__(self, strm):
        self.strm = strm
        self.currTAG = ""
        self.width = None
        self.height = None
        self.box = None

    def startElement(self, name, attrs):
        self.currTAG = name

    def endElement(self, name):
        self.currTAG = ""

    def characters(self, content):

        if self.currTAG == "width":
            self.width = float(content)
            # print self.width
        elif self.currTAG == "height":
            self.height = float(content)
            # print self.height
        elif self.currTAG == "xmin":
            self.box = list()
            self.box.append(float(content.encode("utf-8")))
        elif self.currTAG == "ymin":
            self.box.append(float(content.encode("utf-8")))
        elif self.currTAG == "xmax":
            self.box.append(float(content.encode("utf-8")))
        elif self.currTAG == "ymax":
            self.box.append(float(content.encode("utf-8")))

            CX = ((self.box[0] + self.box[2]) / 2) / self.width
            CY = ((self.box[1] + self.box[3]) / 2) / self.height
            W = (self.box[2] - self.box[0]) / self.width
            H = (self.box[3] - self.box[1]) / self.height

            # self.strm.write("%d " % 1)  # a place-holder, no specific meaning
            # self.strm.write("%.6f " % self.box[0])
            # self.strm.write("%.6f " % self.box[1])
            # self.strm.write("%.6f " % self.box[2])
            # self.strm.write("%.6f\n" % self.box[3])

            self.strm.write("%d " % 1)  # a place-holder, no specific meaning
            self.strm.write("%.6f " % CX)
            self.strm.write("%.6f " % CY)
            self.strm.write("%.6f " % W)
            self.strm.write("%.6f\n" % H)


def xml2txt(xml_dir, txt_dir):
    xml_list = os.listdir(xml_dir)
    xml_list.sort()

    for file in xml_list:
        xml_name = file.split('/')[-1].split('.xml')[0]
        # print xml_name
        txt = os.path.join(txt_dir, xml_name + '.txt')

        parser = xml.sax.make_parser()
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)

        label = open(txt, 'w')
        Handler = ImageNetHandler(label)
        parser.setContentHandler(Handler)
        parser.parse(os.path.join(xml_dir, file))
        label.close()


def txt(img_dir, train_file, val_file, ext='/home/yz/cde/ProposalYOLO/data/imagenet/images/'):
    imgs = os.listdir(img_dir)
    imgs.sort()
    random.shuffle(imgs)

    train = imgs[:484000]
    validation = imgs[484000:]

    strm = open(train_file, 'w')
    for img in train:
        strm.write('%s\n' % os.path.join(ext, img))
    strm.close()

    strm = open('/home/yeezy/Desktop/coco_imagenet.txt', 'a')
    for img in train:
        strm.write('%s\n' % os.path.join(ext, img))
    strm.close()

    strm = open(val_file, 'w')
    for img in validation:
        strm.write('%s\n' % os.path.join(ext, img))
    strm.close()


def copy_img(txt_dir, img_dir, dst_dir):
    img_lst = os.listdir(txt_dir)
    img_lst.sort()

    for item in img_lst:

        name = item.split('.txt')[0]

        srcfn = os.path.join(img_dir, name + '.JPEG')
        dstfn = os.path.join(dst_dir, name + '.JPEG')

        try:
            shutil.copyfile(srcfn, dstfn)
        except:
            print 'the file: ', srcfn, ' not find.'


def check():
    fn = '/home/yeezy/Dat/ImageNet12/all.txt'
    label_dir = '/home/yeezy/Dat/ImageNet12/labels/'

    boxes_list = list()
    img_fn_list = list()
    box_cnt = 0
    with open(fn, 'r') as imgs:
        for img in imgs:
            src_img = img.split('\n')[0]
            dst_img = src_img.replace('images', 'iimages')
            img_fn = dst_img.split('/')[-1].split('.')[0]

            src_label = os.path.join(label_dir, src_img.split('/')[-1].replace('JPEG', 'txt'))
            dst_label = src_label.replace('labels', 'ilabels')
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

    # (1)
    # xml_dir = '/home/yeezy/Dat/ImageNet12/xmls/'
    # txt_dir = '/home/yeezy/Dat/ImageNet12/labels/'
    #
    # lst = os.listdir(xml_dir)
    # lst.sort()
    #
    # for itm in lst:
    #     sub_dir = os.path.join(xml_dir, itm)
    #     print sub_dir
    #     xml2txt(sub_dir, txt_dir)

    # (2)
    # img_dir  = '/home/yeezy/Dat/ImageNet12/images/'
    # train_file = '/home/yeezy/Desktop/imagenet.txt'
    # val_file = '/home/yeezy/Desktop/imagenet_val.txt'
    # txt(img_dir, train_file, val_file)

    # (3)
    # txt_dir = '/home/yeezy/Dat/ImageNet12/labels/'
    # img_dir = '/home/yeezy/Dat/ImageNet12/Training/'
    # dst_dir = '/home/yeezy/Dat/ImageNet12/images/'
    #
    # copy_img(txt_dir, img_dir, dst_dir)

    # (4)
    check()
