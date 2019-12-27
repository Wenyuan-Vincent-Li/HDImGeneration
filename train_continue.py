from options.train_options import TrainOptions
from Training.train import train
from Training import functions

opt = TrainOptions().parse()
opt.alpha = 0.1
opt.name = 'prostateHD'
opt.scale_factor = 0.87
opt.niter = 200
opt.noise_amp = 1
Gs, reals, NoiseAmp = functions.load_trained_pyramid(opt)

reals = reals + [[1024, 1024]]
train(opt, Gs, NoiseAmp, reals)