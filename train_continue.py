from options.train_options import TrainOptions
from Training.train import train
from Training import functions

opt = TrainOptions().parse()
opt.alpha = 1e-3
opt.name = 'prostateHD'
opt.scale_factor = 0.78
opt.niter = 200
opt.noise_amp = 1e-8
Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
Gs = Gs[:1]

train(opt, Gs, Zs, NoiseAmp, reals)