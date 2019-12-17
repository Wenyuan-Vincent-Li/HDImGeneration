from options.train_options import TrainOptions
from Training.train import train
from Training import functions

opt = TrainOptions().parse()
opt.alpha = 0
opt.name = 'prostateHD'
opt.scale_factor = 0.80
opt.niter = 200
opt.noise_amp = 0.1
Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
Gs = Gs[:1]

train(opt, Gs, Zs, NoiseAmp, reals)