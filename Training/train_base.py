import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from Training import functions
from Training.imresize import imresize
import matplotlib.pyplot as plt
from InputPipeline.DataLoader import CreateDataLoader

def train_single_scale(dataloader,netD,netG,reals,Gs,Zs,in_s,NoiseAmp,opt):
    '''
    :param netD: currD
    :param netG: currG
    :param reals: a list of image pyramid ## TODO: you can just pass image shape here
    :param Gs: list of prev netG
    :param Zs: [[Gaussian noise [1, 3, 36, 36]], ]
    :param in_s: 0-> all zero [1, 3, 26, 26]
    :param NoiseAmp: [] -> [1]
    :param opt: config
    :return:
    '''

    real = reals[opt.scale_num] # find the current level image xn
    opt.nzx = real[0] #+(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real[1] #+(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2) # 5
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2) # 5


    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))
    ## Create transforms for mask to downsampled
    maskTransform = []
    for i in range(opt.scale_num):
        modules = [nn.MaxPool2d(2)] * (i + 1)
        maskTransform.append(nn.Sequential(*modules))

    alpha = opt.alpha # 10
    if alpha > 0: # create and preload the image and label for reconstruction loss
        fixed_data_loader = CreateDataLoader(opt, batchSize=opt.num_images, shuffle=False, fixed=True)
        fixed_dataset = fixed_data_loader.load_data()
        fixed_data = next(iter(fixed_dataset))
        opt.num_images = fixed_data['image'].shape[0]
        print('Create fixed reconstruction image dataset for scale level %d, #training images = %d' % (opt.scale_num, opt.num_images))
        fixed_data['image'] = fixed_data['image'].to(opt.device)
        # fixed_data['label'] = fixed_data['label'].to(opt.device)
        temp = []
        for i in fixed_data['down_scale_label']:
            temp.append(i.to(opt.device))
        fixed_data['down_scale_label'] = temp
        del fixed_data_loader, fixed_dataset


    fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], num_samp = opt.num_images)
    ## Notice that the gererated noise has 3 channels [3, 32, 32]
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device) ## z_opt is now all zero [None, 3, 32, 32]
    z_opt = m_noise(z_opt) ## z_opt is now [None, 3, 42, 42]

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []
    # left = -1

    for epoch in range(opt.niter): # niter = 2000
        if Gs == []:
            # z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], opt.batchSize) # [None, 1, 32, 32] ## Generated Gaussian Noise
            # z_opt = m_noise(z_opt.expand(opt.batchSize,3,opt.nzx,opt.nzy)) # [None, 3, 42, 42]
            z_opt = fixed_noise
            z_opt = m_noise(z_opt)
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], opt.batchSize) # [None, 1, 32, 32]
            noise_ = m_noise(noise_.expand(opt.batchSize,3,opt.nzx,opt.nzy)) # [None, 3, 42, 42]
            ## Noise_: for generated false samples and input them to discriminator
        else:
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], opt.batchSize)
            noise_ = m_noise(noise_.expand(opt.batchSize,3,opt.nzx,opt.nzy)) # scale = 1, noise_ [None, 3, 74, 74]

        for j, data in enumerate(dataloader):
            data['image'] = data['image'].to(opt.device)
            data['label'] = data['label'].to(opt.device)
            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################

            # train with real
            netD.zero_grad()
            output = netD(data['image'], data['label']).to(opt.device) # real [None, 3, 32, 32] -> output [None, 1, 22, 22]
            errD_real = -output.mean() #-a W-GAN loss
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            left, right = (epoch * len(dataloader) + j * opt.batchSize) % opt.num_images, (epoch * len(dataloader)
                                                                            + (j+1) * opt.batchSize) % opt.num_images
            # left = (left + 1) % opt.num_images
            # right = left + 1

            if left < right:
                Zs_prime = [i[left:right, :, :, :] for i in Zs]
                fixed_data_label = fixed_data['down_scale_label']
                fixed_data_label = [i[left:right, :, :, :] for i in fixed_data_label]
                z_opt_temp = z_opt[left:right, :, :, :]
                fixed_image_temp = fixed_data['image'][left:right, :, :, :]
                fixed_mask_temp_org = fixed_data['label'][left:right, :, :, :]
                fixed_mask_temp = m_image(fixed_mask_temp_org)
            else:
                z_opt_temp = torch.cat((z_opt[left:, :, :, :], z_opt[:right, :, :, :]), dim=0)
                fixed_image_temp = torch.cat(
                    (fixed_data['image'][left:, :, :, :], fixed_data['image'][:right, :, :, :]), dim=0)
                fixed_mask_temp_org = torch.cat((fixed_data['label'][left:, :, :, :], fixed_data['label'][:right, :, :, :]),
                                            dim=0)
                fixed_mask_temp = m_image(fixed_mask_temp_org)
                Zs_prime = []
                fixed_data_label_temp = fixed_data['down_scale_label']
                fixed_data_label = []
                for idx, value in enumerate(Zs):
                    Zs_prime.append(torch.cat((value[left:,:,:,:], value[:right,:,:,:]), dim = 0))
                    fixed_data_label.append(torch.cat((fixed_data_label_temp[idx][left:,:,:,:], fixed_data_label_temp[idx][:right,:,:,:]), dim = 0))
            # print('Zs_prime', Zs_prime[0].shape)
            # train with fake
            if (j==0) & (epoch == 0): # first iteration training in this level
                if Gs == []:
                    prev = torch.full([opt.batchSize,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    in_s = prev # full of 0 [None, 3, 32, 32]
                    prev = m_image(prev) #[None, 3, 42, 42]
                    # z_prev = torch.full([opt.batchSize,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_prev = torch.full([right - left,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_prev = m_noise(z_prev) # [None, 3, 42, 42]
                    mask = m_image(data['label']) # [None, 3, 42, 42]
                    opt.noise_amp = 1
                else:
                    prev = draw_concat(Gs, Zs_prime, data['down_scale_label'], reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                    ## given a new noise, prev is a image generated by previous Generator with bilinear upsampling [1, 3, 33, 33]

                    z_prev = draw_concat(Gs, Zs_prime, fixed_data_label, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt) ## [1, 3, 33, 33]
                    ## z_prev is a image generated using a specific set of z group, for the purpose determine the intensity of noise
                    criterion = nn.MSELoss()
                    # RMSE = torch.sqrt(criterion(data['image'], z_prev))
                    RMSE = torch.sqrt(criterion(data['image'], prev))
                    opt.noise_amp = opt.noise_amp_init*RMSE
                    z_prev = m_image(z_prev) ## [1, 3, 43, 43]
                    prev = m_image(prev)  ## [1, 3, 43, 43]
                    mask = m_image(data['label'])
            else:
                z_prev = draw_concat(Gs, Zs_prime, fixed_data_label, reals, NoiseAmp, in_s, 'rec', m_noise,
                                     m_image, opt)  ## [1, 3, 33, 33]
                z_prev = m_image(z_prev)
                prev = draw_concat(Gs, Zs_prime, data['down_scale_label'], reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                ## Sample another image generated by the previous generator
                prev = m_image(prev)
                mask = m_image(data['label'])

            if Gs == []:
                noise = noise_ ## Gausiaan noise for generating image [None, 3, 42, 42]
            else:
                noise = opt.noise_amp * noise_+ prev ## [None, 3, 43, 43] new noise is equal to the prev generated image plus the gaussian noise.

            fake = netG(noise.detach(), prev, mask) # [None, 3, 32, 32] the same size with the input image
            # detach() make sure that the gradients don't go to the noise.
            # prev:[None, 3, 42, 42] -> [None, 3, 43, 43] first step prev = 0, second step prev = a image generated by previous Generator with bilinaer upsampling
            output = netD(fake.detach(), data['label']) # output shape [1, 1, 16, 16] -> [1, 1, 23, 23]
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            if alpha > 0:
                Z_opt = opt.noise_amp * z_opt_temp + z_prev
                fake_rec = netG(Z_opt.detach(), z_prev, fixed_mask_temp)
                output_rec = netD(fake_rec.detach(), fixed_mask_temp_org)
                errD_fake_rec = output_rec.mean()
                errD_fake_rec.backward(retain_graph=True)

            gradient_penalty = functions.calc_gradient_penalty(netD, data['image'], fake, data['label'], opt.lambda_grad)
            gradient_penalty.backward()
            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

            errD2plot.append(errD.detach()) ## errD for each iteration


            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################
            netG.zero_grad()
            output = netD(fake, data['label'])
            #D_fake_map = output.detach()
            errG_GAN = -output.mean() # Generator want to make output as large as possible.
            # errG.backward(retain_graph=True)
            
            if alpha!=0: ## alpha = 10 calculate the reconstruction loss
                # output_rec = netD(fake_rec, fixed_mask_temp_org)
                # errG_GAN_rec = -output_rec.mean()
                loss = nn.MSELoss()
                # Z_opt = opt.noise_amp * z_opt_temp + z_prev
#                 print('z_opt_temp', z_opt_temp.shape, 'z_prev', z_prev.shape, 'fixed_mask_temp', fixed_mask_temp.shape,
#                      'fixed_image_temp', fixed_image_temp.shape)
                ## for the first scale z_prev = 0, z_opt = gausian noise, opt.noise_amp = 1 [1, 3, 36, 36]
                ## for the second scale z_prev image generated by Gn, z_opt is all zeros [1, 3, 43, 43]
                # rec_loss = alpha * loss(netG(Z_opt.detach(), z_prev, fixed_mask_temp), fixed_image_temp)
                # rec_loss.backward(retain_graph = True)
                rec_loss = alpha * loss(fake, data['image'])
                errG = errG_GAN + rec_loss
                errG.backward()
                rec_loss = rec_loss.detach()
            else:
                output_rec = netD(fake_rec, fixed_mask_temp_org)
                errG_GAN_rec = -output_rec.mean()
                Z_opt = z_opt_temp
                rec_loss = 0
                errG = errG_GAN + errG_GAN_rec
                errG.backward()
            optimizerG.step()

        ## for every epoch, do the following:
        errG2plot.append(errG.detach()+rec_loss) ## ErrG for each iteration
        D_real2plot.append(D_x) ##  discriminator loss on real
        D_fake2plot.append(D_G_z) ## discriminator loss on fake
        z_opt2plot.append(rec_loss) ## reconstruction loss
        if epoch % 25 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (opt.scale_num, epoch, opt.niter))

        if epoch % 25 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample_%d.png' %  (opt.outf, epoch),
                       functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            plt.imsave('%s/fake_sample_real_%d.png' % (opt.outf, epoch),
                       functions.convert_image_np(data['image']), vmin=0, vmax=1)
            plt.imsave('%s/fake_sample_mask_%d.png' % (opt.outf, epoch),
                       functions.convert_mask_np(data['label']))

            plt.imsave('%s/G(z_opt)_%d.png'    % (opt.outf, epoch),
                       functions.convert_image_np(netG(Z_opt.detach(), z_prev, fixed_mask_temp).detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt)_orgim_%d.png' % (opt.outf, epoch),
                       functions.convert_image_np(fixed_image_temp), vmin=0,
                       vmax=1)
            plt.imsave('%s/G(z_opt)_mask_%d.png' % (opt.outf, epoch),
                       functions.convert_mask_np(fixed_mask_temp))
            #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG,netD,z_opt,opt) ## save netG, netD, z_opt, opt is used to parser output path
    return z_opt,in_s,netG
    # first scale z_opt: generated Gaussion noise [1, 3, 36, 36], in_s = all 0 [1, 3, 26, 26] netG: generator
    # second scale z_opt: all zeros [1, 3, 43, 43], in_s = [1, 3, 26, 26], all zeros

def draw_concat(Gs,Zs,masks,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    '''
    :param Gs: [G0]
    :param Zs: [[Gaussion Noise (1, 3, 36, 36)]]
    :param mask: [down scaled _mask]
    :param reals: [image pyramid] only used to represent the image shape
    :param NoiseAmp: [1]
    :param in_s: all zeros [1, 3, 26, 26]
    :param mode: 'rand'
    :param m_noise:
    :param m_image:
    :param opt:
    :return:
    '''
    if mode == 'rec':
        G_z = in_s[0:1,:,:,:]
    else:
        G_z = in_s[:opt.batchSize,:,:,:] #[None, 3, 26, 26] all zeros, image input for the corest level
    # G_z = in_s[:opt.batchSize, :, :, :]  # [None, 3, 26, 26] all zeros, image input for the corest level
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            for G,Z_opt, mask, real_curr,real_next,noise_amp in zip(Gs,Zs,masks,reals,reals[1:],NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], opt.batchSize)
                    z = z.expand(opt.batchSize, 3, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], opt.batchSize)
                ## z [None, 3, 32, 32]
                z = m_noise(z) ## z [1, 3, 42, 42]
                G_z = G_z[:,:,0:real_curr[0],0:real_curr[1]] ## G_z [None, 3, 32, 32]
                G_z = m_image(G_z) ## G_z [None, 3, 42, 42] all zeros
                z_in = noise_amp*z+G_z ## [None, 3, 42, 42] Gaussian noise

                ## TODO: check the size of noise and mask
                ## resize mask to current level
                mask = m_image(mask)
                G_z = G(z_in.detach(),G_z, mask) ## [1, 3, 26, 26] output of previous generator
                G_z = imresize(G_z,1/opt.scale_factor,opt) ## output upsampling (bilinear) [1, 3, 34, 34]
                G_z = G_z[:,:,0:real_next[0],0:real_next[1]] ## resize the image to be compatible with current G [1, 3, 33, 33]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,mask,real_curr,real_next,noise_amp in zip(Gs,Zs,masks, reals,reals[1:],NoiseAmp):
                Z_opt = Z_opt[:opt.batchSize,:,:,:]
                G_z = G_z[:, :, 0:real_curr[0], 0:real_curr[1]] ## [1, 3, 26, 26] all zeros
                G_z = m_image(G_z) ## [None, 3, 36, 36] all zeros
                z_in = noise_amp*Z_opt+G_z  ## [1, 3, 36, 36] @ scale 1, it's scale 0's fixed gaussian
                mask = m_image(mask)
                G_z = G(z_in.detach(),G_z, mask) ## [1, 3, 26, 26]
                G_z = imresize(G_z,1/opt.scale_factor,opt) ## [1, 3, 34, 34]
                G_z = G_z[:,:,0:real_next[0],0:real_next[1]]  ## [1, 3, 33, 33]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z
