from options.train_options import TrainOptions
from Training.train import train
from Training import functions

opt = TrainOptions().parse()
opt.alpha = 0 ## Not to use reconstruction loss
opt.name = 'prostateHD'
opt.scale_factor = 0.5
Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)

print(len(NoiseAmp), NoiseAmp)
# train(opt, Gs, Zs, NoiseAmp, reals)