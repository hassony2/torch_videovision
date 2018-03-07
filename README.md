torch_videovision - Basic Video transforms for Pytorch
======================================================


This repository implements several basic data-augmentation transforms for pytorch video inputs 

![transform_image](data/transform_cat.png)

The idea was to produce the equivalent of [torchvision transforms](https://github.com/pytorch/vision/tree/master/torchvision/transforms) for video inputs. (The code is therefore widely based on the code from this repository :) ) 

The basic paradigm is that dataloading should produce videoclips as a **list of PIL Images or numpy.ndarrays** (in format as read by opencv).

Several transforms are then provided in [video_transforms](videotransforms/video_transforms.py) that expect such inputs.

Each transform iterates on all the images in the list and applies the wanted augmentation.
So far the following utilities are provided :
- ColorJitter (acts on brightness, saturation, contrast and hue, only on PIL Images for now)
- RandomCrop
- RandomHorizontalFlip
- RandomResize
- RandomRotation

- Resize
- CenterCrop

We then have to convert those inputs to torch tensors.
This can be produced by the [volume_transform](videotransforms/volume_transforms.py).**ClipToTensor** class, which produces a video volume in format (n_channels, n_images, height, width) where n_channels = 3 in case of images.

When randomness is involved, the same random parameters (crop size, scale size, rotation angle,...) are applied to all the frames.

Transforms can be composed just as in torchvision with video_transforms.Compose.

To quickly see a demo of the transformations, run

`python testtransforms.py`

This should produce something like the following image top image (this is a dummy clip for now, so the same image is repeated several times)

On the first row you have the original clip (visualized frame by frame), and on the second row, the transformed one.
