from options.train_options import TrainOptions
from Training.train_mask import train

opt = TrainOptions().parse()
opt.name = 'mask'
Gs = []
Zs = []
NoiseAmp = []
# reals = []
# reals = [[64, 64], [128, 128], [192, 192], [256, 256], [320, 320], [384, 384], [448, 448], [512, 512]]
reals = [[256, 256], [320, 320], [384, 384], [448, 448], [512,512]]
# reals = [[1024,1024]]
# reals = functions.create_reals_pyramid([opt.fineSize, opt.fineSize], reals, opt)

opt.alpha = 0.1
opt.niter = 200
opt.scale_factor = 1.01
opt.noise_amp = 1
opt.no_vgg_loss = True

train(opt, Gs, Zs, NoiseAmp, reals)
