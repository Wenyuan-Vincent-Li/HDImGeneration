import torch
from options.train_options import TrainOptions

from Models.mask_pix2pixHD2 import *
opt = TrainOptions().parse()
netD, netG = init_models(opt)
w = h = 64
noise = torch.rand(4, 4, w, h).to(opt.device)
mask = torch.rand(4, 4, w, h).to(opt.device)
print(mask)
argmax = torch.argmax(mask, dim = 1, keepdim = True)
print(argmax.shape, argmax)
# output = netG(noise, mask)
# output = netD(mask)

# print(output.shape)

# print(len(output), len(output[0]))