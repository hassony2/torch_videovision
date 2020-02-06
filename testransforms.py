import argparse

from matplotlib import pyplot as plt
from PIL import Image

from torchvideotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, Normalize
from torchvideotransforms.volume_transforms import ClipToTensor

img_path = 'data/cat/cat1.jpeg'

parser = argparse.ArgumentParser()
parser.add_argument('--brightness', type=float, default=0.5, help='in [0, 1]')
parser.add_argument('--contrast', type=float, default=0.5, help='in [0, 1]')
parser.add_argument('--hue', type=float, default=0.25, help='in [0, 0.5]')
parser.add_argument('--saturation', type=float, default=0.5, help='in [0, 1]')
parser.add_argument(
    '--rotation_angle',
    type=int,
    default=30,
    help='Max rotation angle in degrees')
parser.add_argument(
    '--scale_size',
    type=int,
    default=256,
    help='Scale smallest clip size to scale_size')
parser.add_argument(
    '--clip_size',
    type=int,
    default=5,
    help='Temporal extent of clip (in number of frames)')
args = parser.parse_args()

# Set transform parameters
crop_size = (200, 200)
channel_nb = 3
clip_size = 5

# Initialize transforms
video_transform_list = [
    RandomRotation(args.rotation_angle),
    Resize(args.scale_size),
    RandomCrop(crop_size),
    ColorJitter(args.brightness, args.contrast, args.saturation, args.hue),
    ClipToTensor(channel_nb=channel_nb),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
