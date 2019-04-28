import os
import xml.sax


class VOCHandler(xml.sax.ContentHandler):

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
        Handler = VOCHandler(label)
        parser.setContentHandler(Handler)
        parser.parse(os.path.join(xml_dir, file))
        label.close()


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

    # xml_dir = '/home/yeezy/Dat/VOC/VOCtest07/Annotations/'
    # txt_dir = '/home/yeezy/Dat/VOC/VOCtest07/Labels/'
    # xml2txt(xml_dir, txt_dir)

    # img_dir  = '/home/yeezy/Dat/VOC/VOCtest07/JPEGImages/'
    # txt_file = '/home/yeezy/Desktop/5k.txt'
    # txt(img_dir, txt_file)

    # img_dir = '/home/yeezy/Dat/ImageNet12/images'
    # txt_file = '/home/yeezy/Desktop/train.txt'
    # txt(img_dir, txt_file)

    # label_dir = '/home/yeezy/Dat/COCO/coco17/labels/val2017'
    # txt_file = '/home/yeezy/Desktop/5k.txt'
    # xtxt(label_dir, txt_file)

    # Megvii
    # label_dir = '/home/yz/cde/ProposalYOLO/data/megvii/labels/val/'
    # txt_file = '/home/yz/cde/ProposalYOLO/data/megvii/val.txt'
    # xtxt(label_dir, txt_file)

    # label_dir = '/home/yz/cde/ProposalYOLO/data/megvii/labels/train/'
    # txt_file = '/home/yz/cde/ProposalYOLO/data/megvii/train.txt'
    # xtxt(label_dir, txt_file)

    # Tiny Megvii
    label_dir = '/home/yz/cde/ProposalYOLO/data/tiny_megvii/labels/val/'
    txt_file = '/home/yz/cde/ProposalYOLO/data/tiny_megvii/val.txt'
    xtxt(label_dir, txt_file)

    label_dir = '/home/yz/cde/ProposalYOLO/data/tiny_megvii/labels/train/'
    txt_file = '/home/yz/cde/ProposalYOLO/data/tiny_megvii/train.txt'
    xtxt(label_dir, txt_file)