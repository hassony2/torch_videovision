import numpy as np
from PIL import Image
import torch


class ToTensor(object):
    """Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]
    """

    def __init__(self, channel_nb=3):
        self.channel_nb = channel_nb

    def __call__(self, clip):
        """
        Args: clip (list of numpy.ndarray): clip (list of images)
        to be converted to tensor.
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, 'Got {0} instead of 3 channels'.format(
                ch)
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image\
            but got list of {0}'.format(type(clip[0])))

        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image\
                but got list of {0}'.format(type(clip[0])))
            img = self.convert_img(img)
            np_clip[:, img_idx, :, :] = img
        tensor_clip = torch.from_numpy(np_clip)
        return tensor_clip.float().div(255)

    def convert_img(self, img):
        """Converts (H, W, C) numpy.ndarray to (C, W, H) format
        """
        img = img.transpose(2, 0, 1)
        return img
