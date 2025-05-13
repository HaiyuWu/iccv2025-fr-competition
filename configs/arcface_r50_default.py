from easydict import EasyDict

config = EasyDict()

config.prefix = "arcface-r50-default"  # make changes here
config.head = "arcface"
config.input_size = [112, 112]
config.embedding_size = 512
config.depth = "50"
config.batch_size = 128
config.weight_decay = 5e-4
config.lr = 0.1
config.momentum = 0.9
config.epochs = 26
config.margin = 0.5
config.fp16 = True
config.sample_rate = 1.0
config.reduce_lr = [12, 20, 24]
config.val_source = "./test_set_package_5" # make changes here
config.train_source = "path/to/.txt/file"  # make changes here
config.val_list = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"] # make changes here
config.augment = True
config.mode = "se"
config.rand_erase = True
