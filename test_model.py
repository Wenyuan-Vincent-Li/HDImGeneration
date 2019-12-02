import torch
class config():
    nc_im = 3
    nc_mask = 4
    nfc = 32
    ker_size = 3
    padd_size = 0
    min_nfc = 16
    num_layer = 5
    device = torch.device('cpu')

from Models.model import *
opt = config()
netG = GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
netG.apply(weights_init)
netD = WDiscriminator(opt).to(opt.device)
netD.apply(weights_init)

x = torch.rand(4, 3, 128, 128)
y = torch.rand(4, 3, 128, 128)
mask = torch.rand(4, 4, 128, 128)

output = netD(x, mask)



print(output.shape)