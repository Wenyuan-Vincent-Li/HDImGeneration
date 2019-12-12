import torch
from options.train_options import TrainOptions

from Models.pix2pixHD import *
opt = TrainOptions().parse()
netD, netG = init_models(opt)
w = h = 64
x = torch.rand(4, 3, w, h).to(opt.device)
y = torch.rand(4, 3, w, h).to(opt.device)
mask = torch.rand(4, 4, w, h).to(opt.device)
output = netG(x, y, mask)
# output = netD(x, mask)


print(output.shape)