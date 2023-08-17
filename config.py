from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    root = '/root/pytorch-YOLO-v1/setup/'
    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 0
    test_num_workers = 4
    batch_size = 8


    # param for optimizer
    lr_ = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    # training
    epoch = 40

    # debug
    debug_file = '/tmp/debugf'

    # model
    load_model_path = None


opt = Config()
