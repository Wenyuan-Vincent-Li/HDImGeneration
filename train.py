from options.train_options import TrainOptions
from Training.train import train

opt = TrainOptions().parse()
opt.name = 'prostateHD'
# opt.dataroot = './Datasets/ColonPair_Fine/'
opt.label_nc = 4
Gs = []
NoiseAmp = []

opt.reals = [[64, 64], [128, 128], [256, 256], [512, 512]]
# opt.reals = [[32, 32], [48, 48], [64, 64]]
opt.alpha = 0.1
opt.niter = 50
opt.scale_factor = 4.00
opt.noise_amp = 1
opt.stop_scale = len(opt.reals)

train(opt, Gs, NoiseAmp, opt.reals)
