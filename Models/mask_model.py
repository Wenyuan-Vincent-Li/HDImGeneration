import torch
from Models.model_base import *

class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.opt = opt
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)  # 32
        self.head = ConvBlock(opt.label_nc, N, opt.ker_size, opt.padd_size,
                              1)  # ConvBlock(in = 3, out = 32, ker = 3, pad = 0, stride = 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        if x.shape[1] == 1:
            x = mask2onehot(x, self.opt.label_nc)
        x = self.head(x)  # [1, 3, 26, 26] -> [1, 32, 24, 24]
        x = self.body(x)  # -> [1, 32, 18, 18]
        x = self.tail(x)  # -> [1, 1, 16, 16]
        return x  # Notice that the output is a feature map not a scalar.


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.opt = opt
        N = opt.nfc
        self.head = ConvBlock(opt.label_nc, N, opt.ker_size, opt.padd_size,
                              1)  # ConvBlock(in = 3, out = 32, ker = 3, pad = 0, stride = 1)
        self.body = nn.Sequential()
        # 3-layer conv block TODO: make it as residuel block?
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size,
                              1)  # ConvBlock(in = 32, out = 32, ker = 3, pad = 0, stride = 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.label_nc, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            # # ConvBlock(in = 32, out = 3, ker = 3, pad = 0, stride = 1)
            nn.Softmax2d()  # map the output to 0 1
        )

    def forward(self, x, y):
        '''
        :param x: noise layer: z vectors
        :param y: prev mask: previous generated mask
        :return:
        '''
        # TODO: make generator conditional in the middle of the network
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y  # Note that the final output is sum

def mask2onehot(label_map, label_nc):
    size = label_map.size()
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
    return input_label


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