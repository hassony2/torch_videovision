import cv2
import random
import numpy as np
import PIL
import torch
from skimage.transform import resize


class Compose(object):
    """Composes several transforms

    Args:
        transforms (list of ``Transform`` objects): list of transforms
        to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


class RandomHorizontalFlip(object):
    """Horizontally flip the list of given images randomly
    with a probability 0.5
    """

    def __call__(self, clip):
        """
        Args:
            img (PIL.Image or numpy.ndarray): List of images to be cropped
                in format (h, w, c) in numpy.ndarray

        Returns:
            PIL.Image or numpy.ndarray: Randomly flipped clip
        """
        if random.random() < 0.5:
            if isinstance(clip[0], np.ndarray):
                return [np.fliplr(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                return [
                    img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip
                ]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image\
                but got list of {0}'.format(type(clip[0])))
        return clip


class Scale(object):
    """Scales a list of (H x W x C) numpy.ndarray to the final size

    The larger the original image is, the more times it takes to
    interpolate

    Args:
        interpolation (str): Can be one of 'nearest', 'bilinear'
            defaults to nearest
    """

    def __init__(self, size, interpolation='nearest'):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        if isinstance(clip[0], np.ndarray):
            if self.interpolation == 'bilinear':
                np_inter = cv2.INTER_LINEAR
            else:
                np_inter = cv2.INTER_NEAREST
            scaled = [
                cv2.resize(img, self.size, interpolation=np_inter)
                for img in clip
            ]
        elif isinstance(clip[0], PIL.Image.Image):
            if self.interpolation == 'bilinear':
                pil_inter = PIL.Image.NEAREST
            else:
                pil_inter = PIL.Image.BILINEAR
            scaled = [img.resize(self.size, pil_inter) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image\
            but got list of {0}'.format(type(clip[0])))
        return scaled


class RandomCrop(object):
    """Extract random crop at the same location for a list of images

    Args:
        size (sequence or int): Desired output size for the
            crop in format (h, w)
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        """
        Args:
            img (PIL.Image or numpy.ndarray): List of images to be cropped
                in format (h, w, c) in numpy.ndarray

        Returns:
            PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image\
            but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            raise (ValueError(
                'Initial image size should be larger then cropped size\
                but got cropped sizes : ({w}, {h})\
                while initial image is ({im_w}, {im_h})'
                .format(im_w=im_w, im_h=im_h, w=w, h=h)))

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)
        if isinstance(clip[0], np.ndarray):
            cropped = [img[y1:y1 + h, x1:x1 + w, :] for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            cropped = [img.crop((x1, y1, x1 + w, y1 + h)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image\
            but got list of {0}'.format(type(clip[0])))
        return cropped
