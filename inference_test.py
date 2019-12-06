from Testing.manipulate import *
from options.train_options import TrainOptions
from Training import functions

opt = TrainOptions().parse()
opt.alpha = 0 ## Not to use reconstruction loss

Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
opt.reals = reals
Gs = Gs[:1]


SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)