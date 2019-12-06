from options.train_options import TrainOptions
from Training.train import train
from Training import functions

opt = TrainOptions().parse()
opt.alpha = 5 ## Not to use reconstruction loss

Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)

train(opt, Gs, Zs, NoiseAmp, reals)