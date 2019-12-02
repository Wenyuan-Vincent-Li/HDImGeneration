import torch
import torch.nn as nn
import numpy as np

def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    if type == 'uniform+poisson':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise2 = np.random.poisson(10, [num_samp, int(size[0]), int(size[1]), int(size[2])])
        noise2 = torch.from_numpy(noise2).to(device)
        noise2 = noise2.type(torch.cuda.FloatTensor)
        noise = noise1+noise2
    if type == 'poisson':
        noise = np.random.poisson(0.1, [num_samp, int(size[0]), int(size[1]), int(size[2])])
        noise = torch.from_numpy(noise).to(device)
        noise = noise.type(torch.cuda.FloatTensor)
    return noise

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)