import torch
from Training import functions
from Training.imresize import imresize
from InputPipeline.DataLoader import CreateDataLoader
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np

def SinGAN_generate(Gs, reals, masks, NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=10):
    if in_s is None:
        in_s = torch.full((num_samples, 3, reals[0][0], reals[0][1]), 0, device=opt.device)
    I_prev = None
    images_gen = []

    for G,real,noise_amp, mask in zip(Gs,opt.reals,NoiseAmp, masks):
        # pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        pad1 = 0
        m = nn.ZeroPad2d(int(pad1))
        nzx = (real[0]-pad1*2)*scale_v
        nzy = (real[1]-pad1*2)*scale_h
        mask = m(mask)
        # images_prev = images_cur
        # images_cur = []
        if n == 0:
            z_curr = functions.generate_noise([1, nzx, nzy], num_samples)
            z_curr = z_curr.expand(num_samples, 3, z_curr.shape[2], z_curr.shape[3])
            z_curr = m(z_curr)
        else:
            z_curr = functions.generate_noise([opt.nc_z, nzx, nzy])
            z_curr = m(z_curr)

        if I_prev is None:
            I_prev = m(in_s)
        else:
            # I_prev = imresize(I_prev, 1 / opt.scale_factor, opt)
            I_prev = I_prev[:, :, 0:round(scale_v * reals[n][0]), 0:round(scale_h * reals[n][1])]
            I_prev = m(I_prev)
            I_prev = I_prev[:, :, 0:z_curr.shape[2], 0:z_curr.shape[3]]
            I_prev = functions.upsampling(I_prev, z_curr.shape[2], z_curr.shape[3])

        # for i in range(0,num_samples,1): ## generate the image one by one
        #     if n == 0:
        #         z_curr = functions.generate_noise([1,nzx,nzy])
        #         z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
        #         z_curr = m(z_curr)
        #     else:
        #         z_curr = functions.generate_noise([opt.nc_z,nzx,nzy])
        #         z_curr = m(z_curr)
        #
        #     if images_prev == []:
        #         I_prev = m(in_s)
        #     else:
        #         I_prev = images_prev[i]
        #         I_prev = imresize(I_prev,1/opt.scale_factor, opt)
        #         I_prev = I_prev[:, :, 0:round(scale_v * reals[n][0]), 0:round(scale_h * reals[n][1])]
        #         I_prev = m(I_prev)
        #         I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
        #         I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])

            # if n < gen_start_scale:
            #     z_curr = Z_opt
        z_in = noise_amp*(z_curr)+I_prev

        I_curr = G(z_in.detach(),I_prev, mask)

            # if n == len(Gs)-1:
            #     if opt.mode == 'train':
            #         dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.name, gen_start_scale)
            #     else:
            #         dir2save = functions.generate_dir2save(opt)
            #     try:
            #         os.makedirs(dir2save)
            #     except OSError:
            #         pass
            #     if (opt.mode != "harmonization") & (opt.mode != "editing") & (opt.mode != "SR") & (opt.mode != "paint2image"):
            #         plt.imsave('%s/%d.png' % (dir2save, i), functions.convert_image_np(I_curr.detach()), vmin=0,vmax=1)
            #         #plt.imsave('%s/%d_%d.png' % (dir2save,i,n),functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
            #         #plt.imsave('%s/in_s.png' % (dir2save), functions.convert_image_np(in_s), vmin=0,vmax=1)
        images_gen.append(I_curr)
        I_prev = I_curr
        n+=1
    return images_gen