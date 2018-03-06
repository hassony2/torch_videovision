import os

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation
from videotransforms.volume_transforms import ClipToTensor

img_path = 'data/cat/cat1.jpeg'

# Set transform parameters
scale_size = (256, 256)
crop_size = (200, 200)
max_rotation_angle = 30
channel_nb = 3
clip_size = 5

# Initialize transforms
video_transform_list = [
    RandomRotation(max_rotation_angle),
    Resize(scale_size),
    RandomCrop(crop_size),
    ClipToTensor(channel_nb=channel_nb)
]
video_transform = Compose(video_transform_list)

# Create dummy video clip
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

# Display resulting image
fig, axes = plt.subplots(2, clip_size)
for img_idx in range(clip_size):
    img = clip[img_idx]
    axes[0, img_idx].imshow(img)
    axes[0, img_idx].axis('off')
    axes[0, 0].set_title('original clip')
    recover_img = tensor_clip.numpy()[:, img_idx, :, :].transpose(1, 2, 0)
    axes[1, img_idx].imshow(recover_img)
    axes[1, img_idx].axis('off')
    axes[1, 0].set_title('transformed clip')
plt.show()
