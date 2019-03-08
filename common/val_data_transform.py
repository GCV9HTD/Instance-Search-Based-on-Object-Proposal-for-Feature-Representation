import cv2
import numpy as np

import torch


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def add(self, transform):
        self.transforms.append(transform)


class ToTensor(object):
    def __init__(self, max_objects=50, is_debug=False):
        self.max_objects = max_objects
        self.is_debug = is_debug

    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        if not self.is_debug:
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))  # channel first
            image = image.astype(np.float32)

        filled_labels = np.zeros((self.max_objects, 4), np.float32)
        filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(filled_labels)}


class KeepAspect(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w, _ = image.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        image_new = np.pad(image, pad, 'constant', constant_values=128)
        padded_h, padded_w, _ = image_new.shape

        # Extract coordinates for unpadded + unscaled image
        x1 = w * (label[:, 0] - label[:, 2]/2)
        y1 = h * (label[:, 1] - label[:, 3]/2)
        x2 = w * (label[:, 0] + label[:, 2]/2)
        y2 = h * (label[:, 1] + label[:, 3]/2)
        # Adjust for added padding
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        # Calculate ratios from coordinates
        label[:, 0] = ((x1 + x2) / 2) / padded_w
        label[:, 1] = ((y1 + y2) / 2) / padded_h
        label[:, 2] *= w / padded_w
        label[:, 3] *= h / padded_h

        return {'image': image_new, 'label': label}


class ResizeImage(object):
    def __init__(self, new_size, interpolation=cv2.INTER_LINEAR):
        self.new_size = tuple(new_size) #  (w, h)
        self.interpolation = interpolation

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = cv2.resize(image, self.new_size, interpolation=self.interpolation)
        return {'image': image, 'label': label}
