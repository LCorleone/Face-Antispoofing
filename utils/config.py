import os


class configuration(object):
    def __init__(self):
        # training setting
        self.NO_GPU = '1'
        self.dataset_name = 'antispoofing'
        self.train_dir = './NN_TRAIN_DATA_merge/train'
        self.val_dir = './NN_TRAIN_DATA_merge/val'
        self.trained_model_dir = './antispoofing_model_resnet152'
        self.batch_size = 16
        self.num_epochs = 10
        # backbone setting
        # resnet n layers
        self.num_layers = 152
        self.embedding_size = 512
        self.image_size = (112, 112)
        self.version_se = 0
        # loss setting
        self.pretrained = ''


CONFIG = configuration()
