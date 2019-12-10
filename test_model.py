import torch
from options.train_options import TrainOptions

from Models.pix2pixHD import *
opt = TrainOptions().parse()
netD, netG = init_models(opt)

x = torch.rand(4, 3, 38, 38).to(opt.device)
y = torch.rand(4, 3, 38, 38).to(opt.device)
mask = torch.rand(4, 4, 38, 38).to(opt.device)

output = netG(x, y, mask)



print(output.shape)