import torch
from utils import *
from InputPipeline.DataLoader import CreateDataLoader
from Models.model import init_models
from Training.train_base import train_single_scale
from Training import functions

def train(opt, Gs, Zs, NoiseAmp):
    in_s = 0
    reals = []
    opt.scale_num = len(Gs)
    opt.reals = functions.create_reals_pyramid([opt.fineSize, opt.fineSize], reals, opt)
    nfc_prev = 0

    while opt.scale_num < opt.stop_scale + 1:
        ## create the dataloader at this scale
        ## TODO: for the draw concat function, you have to imresize the image to lower scale
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        dataset_size = len(data_loader)
        print('Create training dataset for scale level %d, #training images = %d' %(opt.scale_num, dataset_size))

        opt.out_ = functions.generate_dir2save(opt)  # TrainedModels/path_01/scale_factor=0.750000,alpha=10
        opt.outf = '%s/%d' % (opt.out_, opt.scale_num)  # TrainedModels/path_01/scale_factor=0.750000,alpha=10/0

        try:
            os.makedirs(opt.outf)  ## Create a folder to save the image, and trained network
        except OSError:
            pass

        D_curr, G_curr = init_models(opt)

        if (nfc_prev == opt.nfc):
            # @ scale_num = 0 (nfc_prev = 0 != opt.nfc = 32); for level above 1,
            # we reload the trained weights from the last scale
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_, opt.scale_num - 1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, opt.scale_num - 1)))

        z_curr, in_s, G_curr = train_single_scale(dataset, D_curr, G_curr, opt.reals, Gs, Zs, in_s, NoiseAmp, opt)
        # TODO: rethink z_curr, in_s, G_curr meaning, and make sure they are OKAY with each other
        # Think: for the reconstruction loss, different noise should correspond to different images, can you make
        # sure that? we can first try without modification, if it's not working, fix this issue

        ## TODO: figure out in_s role in draw concat

        # G_curr = functions.reset_grads(G_curr, False)
        # ## Change all the requires_grads flage to be False. Aviod data copy for evaluations
        # G_curr.eval()
        # D_curr = functions.reset_grads(D_curr, False)
        # D_curr.eval()
        # TODO: implement eval function and reset_grads

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)  # [1]

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        opt.scale_num += 1
        nfc_prev = opt.nfc  # 32
        del D_curr, G_curr, data_loader, dataset ## TODO; chek if del both data_loader and dataset works
    return