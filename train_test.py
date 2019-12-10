from options.train_options import TrainOptions
from Training.train import train
from Training import functions

opt = TrainOptions().parse()
opt.name = 'prostateHD'
Gs = []
Zs = []
NoiseAmp = []
reals = []
reals = functions.create_reals_pyramid([opt.fineSize, opt.fineSize], reals, opt)

opt.alpha = 10 ## Not to use reconstruction loss
train(opt, Gs, Zs, NoiseAmp, reals)
