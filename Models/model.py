import torch
from Models.model_base import *

class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)  # 32
        self.head = ConvBlock(opt.nc_im + opt.nc_mask, N, opt.ker_size, opt.padd_size,
                              1)  # ConvBlock(in = 3, out = 32, ker = 3, pad = 0, stride = 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x, mask):
        x = torch.cat((x, mask), dim = 1) ## concatenate image and mask
        x = self.head(x)  # [1, 3, 26, 26] -> [1, 32, 24, 24]
        x = self.body(x)  # -> [1, 32, 18, 18]
        x = self.tail(x)  # -> [1, 1, 16, 16]
        return x  # Notice that the output is a feature map not a scalar.


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im + opt.nc_mask, N, opt.ker_size, opt.padd_size,
                              1)  # ConvBlock(in = 3, out = 32, ker = 3, pad = 0, stride = 1)
        self.body = nn.Sequential()
        # 3-layer conv block TODO: make it as residuel block?
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size,
                              1)  # ConvBlock(in = 32, out = 32, ker = 3, pad = 0, stride = 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            # # ConvBlock(in = 32, out = 3, ker = 3, pad = 0, stride = 1)
            nn.Tanh()  # map the output to -1 - 1
        )

    def forward(self, x, y, mask):
        '''
        :param x: noise layer: z vectors
        :param y: prev image: previous generated images
        :param mask: mask with the same resolution as x
        :return:
        '''
        # TODO: make generator conditional in the middle of the network
        ## concatenate the mask and input
        x = torch.cat((x, mask), dim = 1)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y  # Note that the final output is sum

def init_models(opt):
    #generator initialization:
    netG = GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    #discriminator initialization:
    netD = WDiscriminator(opt).to(opt.device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG