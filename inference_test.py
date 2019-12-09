from Testing.manipulate import *
from options.train_options import TrainOptions
from Training import functions
import matplotlib.pyplot as plt

opt = TrainOptions().parse()
opt.alpha = 10 ## Not to use reconstruction loss

Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
opt.reals = reals
# Gs = Gs[:1]

im_gen, mask, orgim = SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)

plt.imsave('test.png',
            functions.convert_mask_np(mask))