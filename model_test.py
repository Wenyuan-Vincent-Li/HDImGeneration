import torch
from options.train_options import TrainOptions
from Models.pix2pixHD2 import *

opt = TrainOptions().parse()
netD, netG = init_models(opt)
w = h = 64
x = torch.rand(4, 3, w, h).to(opt.device)
y = torch.rand(4, 3, w, h).to(opt.device)
mask = torch.rand(4, 4, w, h).to(opt.device)
outputG = netG(x, y, mask)
outputD = netD(x, mask)

print(len(outputD), outputD[0].shape)