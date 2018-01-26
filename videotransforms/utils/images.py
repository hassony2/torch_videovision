def convert_img(self, img):
    """Converts (H, W, C) numpy.ndarray to (C, W, H) format
    """
    img = img.transpose(2, 0, 1)
    return img
