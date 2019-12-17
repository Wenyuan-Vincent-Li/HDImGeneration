import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from Training import functions
from Training.imresize import imresize
import matplotlib.pyplot as plt
from Models.pix2pixHD_base import GANLoss, VGGLoss

class Losses():
    def __init__(self, opt):
        self.criterionGAN = GANLoss(not opt.no_lsgan)
        self.criterionFeat = nn.L1Loss()
        if not opt.no_vgg_loss:
            self.criterionVGG = VGGLoss()


def train_single_scale(dataloader, netD, netG, reals, Gs, Zs, in_s, NoiseAmp, opt):
    '''
    :param netD: currD
    :param netG: currG
    :param reals: a list of image pyramid ## TODO: you can just pass image shape here
    :param Gs: list of prev netG
    :param in_s: 0-> all zero [1, 3, 26, 26]
    :param NoiseAmp: [] -> [1]
    :param opt: config
    :return:
    '''
    loss = Losses(opt)
    real = reals[opt.scale_num]  # find the current level image xn
    opt.nzx = real[0]  # +(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real[1]  # +(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size - 1) * (opt.num_layer - 1)) * opt.stride
    z_opt = 0 ## dummy z_opt
    ## Create transforms for mask to downsampled

    alpha = opt.alpha
    # if alpha > -1:  # create and preload the image and label for reconstruction loss
    #     fixed_data_loader = CreateDataLoader(opt, batchSize=opt.num_images, shuffle=False, fixed=True)
    #     fixed_dataset = fixed_data_loader.load_data()
    #     fixed_data = next(iter(fixed_dataset))
    #     opt.num_images = fixed_data['image'].shape[0]
    #     print('Create fixed reconstruction image dataset for scale level %d, #training images = %d' % (
    #     opt.scale_num, opt.num_images))
    #     fixed_data['image'] = fixed_data['image'].to(opt.device)
    #     fixed_data['label'] = fixed_data['label'].to(opt.device)
    #     temp = []
    #     for i in fixed_data['down_scale_label']:
    #         temp.append(i.to(opt.device))
    #     fixed_data['down_scale_label'] = temp
    #     del fixed_data_loader, fixed_dataset

    # fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], num_samp=opt.num_images)
    ## Notice that the gererated noise has 3 channels [3, 32, 32]
    # z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)  ## z_opt is now all zero [None, 3, 32, 32]
    # z_opt = m_noise(z_opt)  ## z_opt is now [None, 3, 42, 42]

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[opt.niter * 0.8], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[opt.niter * 0.8], gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []

    for epoch in range(opt.niter):  # niter = 2000
        if Gs == []:
            # z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], opt.batchSize) # [None, 1, 32, 32] ## Generated Gaussian Noise
            # z_opt = m_noise(z_opt.expand(opt.batchSize,3,opt.nzx,opt.nzy)) # [None, 3, 42, 42]
            # z_opt = fixed_noise
            # z_opt = m_noise(z_opt)
            noise_ = functions.generate_noise([1, opt.nzx, opt.nzy], opt.batchSize)  # [None, 1, 32, 32]
            noise_ = noise_.expand(opt.batchSize, 3, opt.nzx, opt.nzy)
            ## Noise_: for generated false samples and input them to discriminator
        else:
            noise_ = functions.generate_noise([1, opt.nzx, opt.nzy], opt.batchSize)

        for j, data in enumerate(dataloader):
            data['image'] = data['image'].to(opt.device)
            data['label'] = data['label'].to(opt.device)
            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################

            # train with real
            netD.zero_grad()
            pred_real = netD(data['image'], data['label']) # real [None, 3, 32, 32] -> output [None, 1, 22, 22]
            loss_D_real = loss.criterionGAN(pred_real, True)

            D_x = loss_D_real.item()


            # train with fake
            if (j == 0) & (epoch == 0):  # first iteration training in this level
                if Gs == []:
                    prev = torch.full([opt.batchSize, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    in_s = prev  # full of 0 [None, 3, 32, 32]
                    mask = data['label']
                    # opt.noise_amp = opt.noise_amp_init
                    opt.noise_amp = 1
                else:
                    prev = draw_concat(Gs, data['down_scale_label'], reals, NoiseAmp, in_s, 'rand', opt)
                    ## given a new noise, prev is a image generated by previous Generator with bilinear upsampling [1, 3, 33, 33]

                    # z_prev = draw_concat(Gs, Zs_prime, fixed_data_label, reals, NoiseAmp, in_s, 'rec', m_noise, m_image,
                    #                      opt)  ## [1, 3, 33, 33]
                    ## z_prev is a image generated using a specific set of z group, for the purpose determine the intensity of noise
                    criterion = nn.MSELoss()
                    # RMSE = torch.sqrt(criterion(data['image'], z_prev))
                    RMSE = torch.sqrt(criterion(data['image'], prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    # z_prev = m_image(z_prev)  ## [1, 3, 43, 43]
                    # prev = m_image(prev)  ## [1, 3, 43, 43]
                    # mask = m_image(data['label'])
                    mask = data['label']
            else:
                # z_prev = draw_concat(Gs, Zs_prime, fixed_data_label, reals, NoiseAmp, in_s, 'rec', m_noise,
                #                      m_image, opt)  ## [1, 3, 33, 33]
                # z_prev = m_image(z_prev)
                prev = draw_concat(Gs, data['down_scale_label'], reals, NoiseAmp, in_s, 'rand', opt)
                ## Sample another image generated by the previous generator
                # prev = m_image(prev)
                # mask = m_image(data['label'])
                mask = data['label']

            if Gs == []:
                noise = noise_  ## Gausiaan noise for generating image [None, 3, 42, 42]
            else:
                noise = opt.noise_amp * noise_ + prev  ## [None, 3, 43, 43] new noise is equal to the prev generated image plus the gaussian noise.

            fake = netG(noise.detach(), prev, mask)  # [None, 3, 32, 32] the same size with the input image
            # detach() make sure that the gradients don't go to the noise.
            # prev:[None, 3, 42, 42] -> [None, 3, 43, 43] first step prev = 0, second step prev = a image generated by previous Generator with bilinaer upsampling
            pred_fake = netD(fake.detach(), data['label'])  # output shape [1, 1, 16, 16] -> [1, 1, 23, 23]
            loss_D_fake = loss.criterionGAN(pred_fake, False)
            D_G_z = loss_D_fake.item()

            errD = (loss_D_real + loss_D_fake) * 0.5
            errD.backward()
            optimizerD.step()

            errD2plot.append(errD.detach())  ## errD for each iteration

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################
            netG.zero_grad()
            pred_fake = netD(fake, data['label'])
            loss_G_GAN = loss.criterionGAN(pred_fake, True)

            # GAN feature matching loss
            loss_G_GAN_Feat = 0
            if not opt.no_ganFeat_loss:
                feat_weights = 4.0 / (opt.n_layers_D + 1)
                D_weights = 1.0 / opt.num_D
                for i in range(opt.num_D):
                    for j in range(len(pred_fake[i]) - 1):
                        loss_G_GAN_Feat += D_weights * feat_weights * \
                                           loss.criterionFeat(pred_fake[i][j],
                                                              pred_real[i][j].detach()) * opt.lambda_feat

            # VGG feature matching loss
            loss_G_VGG = 0
            if not opt.no_vgg_loss:
                loss_G_VGG = loss.criterionVGG(fake, data['image']) * opt.lambda_feat

            ## reconstruction loss
            if alpha != 0:  ## alpha = 10 calculate the reconstruction loss
                Recloss = nn.MSELoss()
                rec_loss = alpha * Recloss(fake, data['image'])
            else:
                rec_loss = 0

            errG = loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG + rec_loss
            errG.backward()
            optimizerG.step()


        ## for every epoch, do the following:
        errG2plot.append(errG.detach())  ## ErrG for each iteration
        D_real2plot.append(D_x)  ##  discriminator loss on real
        D_fake2plot.append(D_G_z)  ## discriminator loss on fake

        if epoch % 25 == 0 or epoch == (opt.niter - 1):
            print('scale %d:[%d/%d]' % (opt.scale_num, epoch, opt.niter))

        if epoch % 25 == 0 or epoch == (opt.niter - 1):
            plt.imsave('%s/fake_sample_%d.png' % (opt.outf, epoch),
                       functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            plt.imsave('%s/fake_sample_real_%d.png' % (opt.outf, epoch),
                       functions.convert_image_np(data['image']), vmin=0, vmax=1)
            plt.imsave('%s/fake_sample_mask_%d.png' % (opt.outf, epoch),
                       functions.convert_mask_np(data['label']))
            # plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            # plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            # plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            # plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            # plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            # plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)

            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG, netD, z_opt, opt)  ## save netG, netD, z_opt, opt is used to parser output path
    return z_opt, in_s, netG
    # first scale z_opt: generated Gaussion noise [1, 3, 36, 36], in_s = all 0 [1, 3, 26, 26] netG: generator
    # second scale z_opt: all zeros [1, 3, 43, 43], in_s = [1, 3, 26, 26], all zeros


def draw_concat(Gs, masks, reals, NoiseAmp, in_s, mode, opt):
    '''
    :param Gs: [G0]
    :param mask: [down scaled _mask]
    :param reals: [image pyramid] only used to represent the image shape
    :param NoiseAmp: [1]
    :param in_s: all zeros [1, 3, 26, 26]
    :param mode: 'rand'
    :param opt:
    :return:
    '''
    G_z = in_s[:opt.batchSize, :, :, :]  # [None, 3, 26, 26] all zeros, image input for the corest level
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            # pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
            pad_noise = 0
            for G, mask, real_curr, real_next, noise_amp in zip(Gs, masks, reals, reals[1:], NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, real_curr[0] - 2 * pad_noise, real_curr[1] - 2 * pad_noise],
                                                 opt.batchSize)
                    z = z.expand(opt.batchSize, 3, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise(
                        [opt.nc_z, real_curr[0] - 2 * pad_noise, real_curr[1] - 2 * pad_noise], opt.batchSize)
                G_z = G_z[:, :, 0:real_curr[0], 0:real_curr[1]]  ## G_z [None, 3, 32, 32]
                z_in = noise_amp * z + G_z
                G_z = G(z_in.detach(), G_z, mask)  ## [1, 3, 26, 26] output of previous generator
                # G_z = imresize(G_z, 1 / opt.scale_factor, opt)  ## output upsampling (bilinear) [1, 3, 34, 34]
                G_z = imresize(G_z, real_next[1] / real_curr[1], opt)
                G_z = G_z[:, :, 0:real_next[0],
                      0:real_next[1]]  ## resize the image to be compatible with current G [1, 3, 33, 33]
                count += 1
    return G_z
