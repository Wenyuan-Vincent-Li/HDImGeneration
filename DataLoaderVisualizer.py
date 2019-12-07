if __name__ == "__main__":
    from utils import *
    from options.train_options import TrainOptions
    from InputPipeline.DataLoader import CreateDataLoader
    from Training import functions
    opt = TrainOptions().parse()
    reals = []
    opt.reals = functions.create_reals_pyramid([opt.fineSize, opt.fineSize], reals, opt)
    opt.scale_num = 1
    # data_loader = CreateDataLoader(opt)
    # dataset = data_loader.load_data()
    # dataset_size = len(data_loader)
    # print('#training images = %d' % dataset_size)
    #
    # for idx, data in enumerate(dataset):
    #     if idx == 0:
    #         print(data['label'].shape, data['image'].shape)
    #         break
    # print(data['path'])
    # print(np.unique(data['label']))
    # display_sementic((data['image'][0, ...].numpy() + 1 )/ 2 * 255, data['label'][0,0,...].numpy())
    #
    fixed_data_loader = CreateDataLoader(opt, batchSize=opt.num_images, shuffle=False, fixed=True)
    dataset = fixed_data_loader.load_data()

    data = next(iter(dataset))
    print(data['label'].shape, data['image'].shape)
    # print(data['path'])
    exit()
    print(np.unique(data['label']))
    # display_sementic((data['image'][0, ...].numpy() + 1 )/ 2 * 255, data['label'][0,0,...].numpy())