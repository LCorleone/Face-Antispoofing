import os


class configuration(object):
    def __init__(self):
        # training setting
        self.NO_GPU = '0'
        self.dataset_name = 'antispoofing'
        self.train_dir = './NN_TRAIN_DATA/train'
        self.val_dir = './NN_TRAIN_DATA/val'
        self.trained_model_dir = './antispoofing_model_resnet100_dual'
        self.batch_size = 32
        self.num_epochs = 15
        # backbone setting
        # resnet n layers
        self.num_layers = 100
        self.embedding_size = 512
        self.image_size = (112, 112)
        self.version_se = 0
        # loss setting
        self.pretrained = ''


CONFIG = configuration()
