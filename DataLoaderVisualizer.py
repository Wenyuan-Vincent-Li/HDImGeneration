if __name__ == "__main__":
    from utils import *
    from options.train_options import TrainOptions
    from InputPipeline.DataLoader import CreateDataLoader
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    for idx, data in enumerate(dataset):
        if idx == 0:
            print(data['label'].shape, data['image'].shape)
            break
    print(data['path'])
    display_sementic((data['image'][0, ...].numpy() + 1 )/ 2 * 255, data['label'][0,0,...].numpy())