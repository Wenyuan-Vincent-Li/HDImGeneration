from options.train_options import TrainOptions
from Training.train import train
from Training import functions

opt = TrainOptions().parse()
opt.name = 'prostate'
Gs = []
Zs = []
NoiseAmp = []
reals = []
# reals = [[64, 64], [128, 128], [192, 192], [256, 256], [320, 320], [384, 384], [448, 448], [512, 512]]
reals = functions.create_reals_pyramid([opt.fineSize, opt.fineSize], reals, opt)

opt.alpha = 1 ## Not to use reconstruction loss
train(opt, Gs, Zs, NoiseAmp, reals)
