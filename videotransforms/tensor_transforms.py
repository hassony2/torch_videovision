from videotransforms.utils import functional as F


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation

    Given mean: m and std: s
    will  normalize each channel as channel = (channel - mean) / std

    Args:
        mean (int): mean value
        std (int): std value
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of stacked images or image
            of size (C, H, W) to be normalized

        Returns:
            Tensor: Normalized stack of image of image
        """
        return F.normalize(tensor, self.mean, self.std)
