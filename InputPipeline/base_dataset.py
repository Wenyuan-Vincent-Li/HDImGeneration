import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_params(opt, size):
    w, h = size
    ## random scale:
    new_w = int(random.uniform(opt.fineSize / w, 2 - opt.fineSize / w) * w)
    new_h = new_w * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    horizontal_flip = random.random() > 0.5
    vertical_flip = random.random() > 0.5
    rotation = random.randint(0, 3)
    return {'scale_target': (new_w, new_h), 'crop_pos': (x, y), 'horizontal_flip': horizontal_flip,
            'vertical_flip': vertical_flip, 'rotation': rotation}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    ## First random scale the image
    if opt.randomScale:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, params['scale_target'][0], method)))
    ## Then random crop the image to certain size
    transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __horizontal_flip(img, params['horizontal_flip'])))
        transform_list.append(transforms.Lambda(lambda img: __vertical_flip(img, params['vertical_flip'])))
        transform_list.append(transforms.Lambda(lambda img: __rotate(img, params['rotation'])))
    if opt.isTrain:
        ## Finally scale down to the current levle
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.reals[opt.scale_num][0], method)))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __horizontal_flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __vertical_flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def __rotate(img, rotation):
    return img.rotate(90 * rotation)
