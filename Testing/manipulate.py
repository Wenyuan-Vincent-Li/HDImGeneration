import torch
from Training import functions
from Training.imresize import imresize
from InputPipeline.DataLoader import CreateDataLoader
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np

def SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=50):
    opt.batchSize = 1
    opt.mode = 'train'
    opt.scale_num = len(Gs)
    # opt.scale_num = 2
    # print(opt.scale_num)
    # print(opt.reals[opt.scale_num])
    # exit()

    opt.out = functions.generate_dir2save(opt)  # TrainedModels/path_01/scale_factor=0.750000,alpha=10
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    data = next(iter(dataset))
    masks = data['down_scale_label']


    if in_s is None:
        in_s = torch.full((1, 3, reals[0][0], reals[0][1]), 0, device=opt.device)
        # in_s = imresize(data['image'], pow(opt.scale_factor, len(Gs)), opt)
    images_cur = []


    for G,Z_opt,noise_amp, mask in zip(Gs,Zs,NoiseAmp, masks):
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h
        mask = m(mask)
        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1): ## generate the image one by one
            if n == 0:
                z_curr = functions.generate_noise([1,nzx,nzy])
                z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
                z_curr = m(z_curr)
            else:
                z_curr = functions.generate_noise([opt.nc_z,nzx,nzy])
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)
                #I_prev = m(I_prev)
                #I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                #I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)
                if opt.mode != "SR":
                    I_prev = I_prev[:, :, 0:round(scale_v * reals[n][0]), 0:round(scale_h * reals[n][1])]
                    I_prev = m(I_prev)
                    I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                    I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
                else:
                    I_prev = m(I_prev)

            if n < gen_start_scale:
                z_curr = Z_opt
            z_in = noise_amp*(z_curr)+I_prev

            # if n ==0:
                # print(z_in, z_in.shape, torch.eq(z_in, torch.full(z_curr.shape, 0, device=opt.device)))
                # exit()
            I_curr = G(z_in.detach(),I_prev, mask)

            if n == len(Gs)-1:
                if opt.mode == 'train':
                    dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.name, gen_start_scale)
                else:
                    dir2save = functions.generate_dir2save(opt)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                if (opt.mode != "harmonization") & (opt.mode != "editing") & (opt.mode != "SR") & (opt.mode != "paint2image"):
                    plt.imsave('%s/%d.png' % (dir2save, i), functions.convert_image_np(I_curr.detach()), vmin=0,vmax=1)
                    #plt.imsave('%s/%d_%d.png' % (dir2save,i,n),functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
                    #plt.imsave('%s/in_s.png' % (dir2save), functions.convert_image_np(in_s), vmin=0,vmax=1)
            images_cur.append(I_curr)
        n+=1
    return I_curr.detach(), data['label'], data['image']