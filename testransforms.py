import os

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from videotransforms.video_transforms import Compose, Scale, RandomCrop
from videotransforms.volume_transforms import ToTensor

img_path = 'data/cat/cat1.jpeg'

scale_size = (256, 256)
crop_size = (200, 200)
channel_nb = 3
clip_size = 5

video_transform_list = [
    Scale(scale_size),
    RandomCrop(crop_size),
    ToTensor(channel_nb=channel_nb)
]
video_transform = Compose(video_transform_list)
img = Image.open(img_path)
clip = [img] * clip_size
tensor_clip = video_transform(clip)

tensor_shape = tensor_clip.shape
assert tensor_shape[
    0] == channel_nb, 'Channel dimension should be {} but got {}'.format(
        channel_nb, tensor_shape[0])
assert tensor_shape[
    1] == clip_size, 'Time dimension should be {} but got {}'.format(
        clip_size, tensor_shape[1])
spatial_dim = tuple(tensor_shape[2:4])
assert spatial_dim == crop_size, 'Final tensor spatial dims {} should be crop_size {}'.format(
    spatial_dim, clip_size)

recover_img = tensor_clip.numpy()[:, 2, :, :].transpose(1, 2, 0)
plt.imshow(recover_img)
plt.show()
import pdb
pdb.set_trace()
