from utils import *
from options.train_options import TrainOptions
from InputPipeline.DataLoader import CreateDataLoader
from Models.model import init_models
from Training.train import train
from Training import functions

opt = TrainOptions().parse()
Gs = []
Zs = []
NoiseAmp = []
# opt.alpha = 0 ## Not to use reconstruction loss
train(opt, Gs, Zs, NoiseAmp)