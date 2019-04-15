import keras
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('pdf')
import matplotlib.pylab as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Activation, Add, Dense, Flatten, Dropout, Multiply, Embedding, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, PReLU, Layer, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD, Adam

import os
import keras.backend.tensorflow_backend as KTF
import sys
# sys.path.append('../utils')
# from config import CONFIG


def residual_unit_v3(data, filters, strides, dim_match, name, bottle_neck, **kwargs):
    use_se = kwargs.get('version_se', 0)
    bn_mom = kwargs.get('bn_mom', 0.9)
    if bottle_neck:
        # bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        bn1 = BatchNormalization(
            momentum=bn_mom, epsilon=2e-5, name=name + '_bn1')(data)
        # conv1 = Conv(data=bn1, filters=int(filters*0.25), kernel_size=(1,1), strides=(1,1), pad=(0,0),
        #                            no_bias=True, workspace=workspace, name=name + '_conv1')
        conv1 = Conv2D(int(filters * 0.25), kernel_size=(1, 1), strides=(1, 1),
                       padding='same', use_bias=False, name=name + '_conv1')(bn1)
        # bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        bn2 = BatchNormalization(
            momentum=bn_mom, epsilon=2e-5, name=name + '_bn2')(conv1)
        # act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
        act1 = PReLU(alpha_initializer=keras.initializers.glorot_normal(
        ), name=name + '_relu1')(bn2)
        # conv2 = Conv(data=act1, filters=int(filters*0.25), kernel_size=(3,3), strides=(1,1), pad=(1,1),
        #                            no_bias=True, workspace=workspace, name=name + '_conv2')
        conv2 = Conv2D(int(filters * 0.25), kernel_size=(3, 3), strides=(1, 1),
                       padding='same', use_bias=False, name=name + '_conv2')(act1)
        # bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        bn3 = BatchNormalization(
            momentum=bn_mom, epsilon=2e-5, name=name + '_bn3')(conv2)
        # act2 = Act(data=bn3, act_type=act_type, name=name + '_relu2')
        act2 = PReLU(alpha_initializer=keras.initializers.glorot_normal(
        ), name=name + '_relu2')(bn3)
        # conv3 = Conv(data=act2, filters=filters, kernel_size=(1,1), strides=strides, pad=(0,0), no_bias=True,
        #                            workspace=workspace, name=name + '_conv3')
        conv3 = Conv2D(filters, kernel_size=(1, 1), strides=strides,
                       padding='same', use_bias=False, name=name + '_conv3')(act2)
        # bn4 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4')
        bn4 = BatchNormalization(
            momentum=bn_mom, epsilon=2e-5, name=name + '_bn4')(conv3)
        if use_se:
            # se begin
            # body = mx.sym.Pooling(data=bn4, global_pool=True, kernel_size=(7, 7), pool_type='avg', name=name+'_se_pool1')

            body = AveragePooling2D(pool_size=(int(bn4.shape[1]), int(
                bn4.shape[1])), padding='same', name=name + '_se_pool1')(bn4)
            # body = Conv(data=body, filters=filters//16, kernel_size=(1,1), strides=(1,1), pad=(0,0),
            #                           name=name+"_se_conv1", workspace=workspace)
            body = Conv2D(filters // 16, kernel_size=(1, 1), strides=(1, 1),
                          padding='same', use_bias=False, name=name + '_se_conv1')(body)
            # body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
            body = PReLU(alpha_initializer=keras.initializers.glorot_normal(
            ), name=name + '_se_relu1')(body)
            # body = Conv(data=body, filters=filters, kernel_size=(1,1), strides=(1,1), pad=(0,0),
            #                           name=name+"_se_conv2", workspace=workspace)
            body = Conv2D(filters, kernel_size=(1, 1), strides=(
                1, 1), padding='same', use_bias=False, name=name + '_se_conv2')(body)
            # body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
            body = Activation(activation='sigmoid',
                              name=name + '_se_sigmoid')(body)
            # bn4 = mx.symbol.broadcast_mul(bn4, body)
            bn4 = Multiply()([bn4, body])

            # se end

        if dim_match:
            shortcut = data
        else:
            # conv1sc = Conv(data=data, filters=filters, kernel_size=(1,1), strides=strides, no_bias=True,
            #                                 workspace=workspace, name=name+'_conv1sc')
            conv1sc = Conv2D(filters, kernel_size=(1, 1), strides=strides,
                             padding='same', use_bias=False, name=name + '_conv1sc')(data)
            # shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc')
            shortcut = BatchNormalization(
                momentum=bn_mom, epsilon=2e-5, name=name + '_sc')(conv1sc)
        return Add()([bn4, shortcut])

    else:
        # bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        bn1 = BatchNormalization(
            momentum=bn_mom, epsilon=2e-5, name=name + '_bn1')(data)
        # conv1 = Conv(data=bn1, filters=filters, kernel_size=(3,3), strides=(1,1), pad=(1,1),
        #                               no_bias=True, workspace=workspace, name=name + '_conv1')
        conv1 = Conv2D(filters, kernel_size=(3, 3), strides=(
            1, 1), padding='same', use_bias=False, name=name + '_conv1')(bn1)
        # bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        bn2 = BatchNormalization(
            momentum=bn_mom, epsilon=2e-5, name=name + '_bn2')(conv1)
        # act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
        act1 = PReLU(alpha_initializer=keras.initializers.glorot_normal(
        ), name=name + '_relu1')(bn2)
        # conv2 = Conv(data=act1, filters=filters, kernel_size=(3,3), strides=strides, pad=(1,1),
        #                               no_bias=True, workspace=workspace, name=name + '_conv2')
        conv2 = Conv2D(filters, kernel_size=(3, 3), strides=strides,
                       padding='same', use_bias=False, name=name + '_conv2')(act1)
        # bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        bn3 = BatchNormalization(
            momentum=bn_mom, epsilon=2e-5, name=name + '_bn3')(conv2)
        if use_se:
            # se begin
            # body = mx.sym.Pooling(data=bn3, global_pool=True, kernel_size=(7, 7), pool_type='avg', name=name+'_se_pool1')
            body = AveragePooling2D(pool_size=(int(bn3.shape[1]), int(
                bn3.shape[1])), padding='same', name=name + '_se_pool1')(bn3)
            # body = Conv(data=body, filters=filters//16, kernel_size=(1,1), strides=(1,1), pad=(0,0),
            #                           name=name+"_se_conv1", workspace=workspace)
            body = Conv2D(filters // 16, kernel_size=(1, 1), strides=(1, 1),
                          padding='same', use_bias=False, name=name + '_se_conv1')(body)
            # body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
            body = PReLU(alpha_initializer=keras.initializers.glorot_normal(
            ), name=name + '_se_relu1')(body)
            # body = Conv(data=body, filters=filters, kernel_size=(1,1), strides=(1,1), pad=(0,0),
            #                         name=name+"_se_conv2", workspace=workspace)
            body = Conv2D(filters, kernel_size=(1, 1), strides=(
                1, 1), padding='same', use_bias=False, name=name + '_se_conv2')(body)
            # body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
            body = Activation(activation='sigmoid',
                              name=name + '_se_sigmoid')(body)
            # bn3 = mx.symbol.broadcast_mul(bn3, body)
            bn3 = Multiply()([bn3, body])
            # se end

        if dim_match:
            shortcut = data
        else:
            # conv1sc = Conv(data=data, filters=filters, kernel_size=(1,1), strides=strides, no_bias=True,
            #                                 workspace=workspace, name=name+'_conv1sc')
            conv1sc = Conv2D(filters, kernel_size=(1, 1), strides=strides,
                             padding='same', use_bias=False, name=name + '_conv1sc')(data)
            # shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
            shortcut = BatchNormalization(
                momentum=bn_mom, epsilon=2e-5, name=name + '_sc')(conv1sc)

        return Add()([bn3, shortcut])


def resnet_binary(nn_input_shape, units, num_stages, filter_list, embedding, bottle_neck, **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    num_unit = len(units)
    assert(num_unit == num_stages)

    nn_input = Input(shape=nn_input_shape)

    # body = Conv(data=body, filters=filter_list[0], kernel_size=(3,3), strides=(1,1), pad=(1, 1),
    #                         no_bias=True, name="conv0", workspace=workspace)
    body = Conv2D(filters=filter_list[0], kernel_size=(3, 3), strides=(
        1, 1), padding='same', use_bias=False, name='conv0')(nn_input)
    # body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = BatchNormalization(momentum=bn_mom, epsilon=2e-5, name='bn0')(body)
    # body = Act(data=body, act_type=act_type, name='relu0')
    body = PReLU(
        alpha_initializer=keras.initializers.glorot_normal(), name='relu0')(body)

    for i in range(num_stages):

        body = residual_unit_v3(body, filter_list[i + 1], (2, 2), False, name='stage%d_unit%d' % (
            i + 1, 1), bottle_neck=bottle_neck, **kwargs)
        for j in range(units[i] - 1):
            body = residual_unit_v3(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (
                i + 1, j + 2), bottle_neck=bottle_neck, **kwargs)

    # fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
    # body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    body = BatchNormalization(momentum=bn_mom, epsilon=2e-5, name='bn1')(body)
    # body = mx.symbol.Dropout(data=body, p=0.4)
    body = Dropout(0.4)(body)
    # fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
    body = Flatten()(body)
    fc1 = Dense(embedding, name='pre_fc1')(body)
    # fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
    fc1 = BatchNormalization(scale=False, momentum=bn_mom,
                             epsilon=2e-5, name='fc1')(fc1)
    binary = Dense(1, activation='sigmoid', use_bias=False, name='binary')(fc1)
    model = Model(nn_input, binary)
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])

    return model


def resnet_binary_test(nn_input_shape, units, num_stages, filter_list, embedding, bottle_neck, **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    num_unit = len(units)
    assert(num_unit == num_stages)

    nn_input = Input(shape=nn_input_shape)

    # body = Conv(data=body, filters=filter_list[0], kernel_size=(3,3), strides=(1,1), pad=(1, 1),
    #                         no_bias=True, name="conv0", workspace=workspace)
    body = Conv2D(filters=filter_list[0], kernel_size=(3, 3), strides=(
        1, 1), padding='same', use_bias=False, name='conv0')(nn_input)
    # body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = BatchNormalization(momentum=bn_mom, epsilon=2e-5, name='bn0')(body)
    # body = Act(data=body, act_type=act_type, name='relu0')
    body = PReLU(
        alpha_initializer=keras.initializers.glorot_normal(), name='relu0')(body)

    for i in range(num_stages):

        body = residual_unit_v3(body, filter_list[i + 1], (2, 2), False, name='stage%d_unit%d' % (
            i + 1, 1), bottle_neck=bottle_neck, **kwargs)
        for j in range(units[i] - 1):
            body = residual_unit_v3(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (
                i + 1, j + 2), bottle_neck=bottle_neck, **kwargs)

    # fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
    # body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    body = BatchNormalization(momentum=bn_mom, epsilon=2e-5, name='bn1')(body)
    # body = mx.symbol.Dropout(data=body, p=0.4)
    body = Dropout(0.5)(body)
    # fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
    # body = Flatten()(body)
    # fc1 = Dense(embedding, name='pre_fc1')(body)
    fc1 = GlobalAveragePooling2D()(body)
    # fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
    fc1 = BatchNormalization(scale=False, momentum=bn_mom,
                             epsilon=2e-5, name='fc1')(fc1)
    binary = Dense(1, activation='sigmoid', use_bias=False, name='binary')(fc1)
    model = Model(nn_input, binary)
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])

    return model


def get_symbol_binary(nn_input_shape, num_layers, embedding, **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    if num_layers >= 101:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
    num_stages = 4
    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 49:
        units = [3, 4, 14, 3]
    elif num_layers == 50:
        units = [3, 4, 14, 3]
    elif num_layers == 74:
        units = [3, 6, 24, 3]
    elif num_layers == 90:
        units = [3, 8, 30, 3]
    elif num_layers == 100:
        units = [3, 13, 30, 3]
    elif num_layers == 124:
        units = [3, 13, 40, 5]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 152:
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError(
            "no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    return resnet_binary(nn_input_shape=nn_input_shape, embedding=embedding, units=units, num_stages=num_stages,
                         filter_list=filter_list, bottle_neck=bottle_neck, **kwargs)


def dual_backbone(nn_input_shape, units, num_stages, filter_list, embedding, bottle_neck, **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    num_unit = len(units)
    assert(num_unit == num_stages)

    nn_input = Input(shape=nn_input_shape)

    # body = Conv(data=body, filters=filter_list[0], kernel_size=(3,3), strides=(1,1), pad=(1, 1),
    #                         no_bias=True, name="conv0", workspace=workspace)
    body = Conv2D(filters=filter_list[0], kernel_size=(3, 3), strides=(
        1, 1), padding='same', use_bias=False, name='conv0')(nn_input)
    # body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = BatchNormalization(momentum=bn_mom, epsilon=2e-5, name='bn0')(body)
    # body = Act(data=body, act_type=act_type, name='relu0')
    body = PReLU(
        alpha_initializer=keras.initializers.glorot_normal(), name='relu0')(body)

    for i in range(num_stages):

        body = residual_unit_v3(body, filter_list[i + 1], (2, 2), False, name='stage%d_unit%d' % (
            i + 1, 1), bottle_neck=bottle_neck, **kwargs)
        for j in range(units[i] - 1):
            body = residual_unit_v3(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (
                i + 1, j + 2), bottle_neck=bottle_neck, **kwargs)

    # fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
    # body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    body = BatchNormalization(momentum=bn_mom, epsilon=2e-5, name='bn1')(body)
    # body = mx.symbol.Dropout(data=body, p=0.4)
    body = Dropout(0.4)(body)
    # fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
    body = Flatten()(body)
    fc1 = Dense(embedding, name='pre_fc1')(body)
    # fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
    fc1 = BatchNormalization(scale=False, momentum=bn_mom,
                             epsilon=2e-5, name='fc1')(fc1)
    binary = Dense(1, activation='sigmoid', use_bias=False, name='binary')(fc1)
    model = Model(nn_input, binary)
    sgd = SGD(lr=0.01, momentum=0.9, decay=1.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def get_symbol_dual(nn_input_shape, num_layers, embedding, **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    if num_layers >= 101:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
    num_stages = 4
    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 49:
        units = [3, 4, 14, 3]
    elif num_layers == 50:
        units = [3, 4, 14, 3]
    elif num_layers == 74:
        units = [3, 6, 24, 3]
    elif num_layers == 90:
        units = [3, 8, 30, 3]
    elif num_layers == 100:
        units = [3, 13, 30, 3]
    elif num_layers == 124:
        units = [3, 13, 40, 5]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 152:
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError(
            "no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    return dual_backbone(nn_input_shape=nn_input_shape, embedding=embedding, units=units, num_stages=num_stages,
                         filter_list=filter_list, bottle_neck=bottle_neck, **kwargs)


if __name__ == '__main__':
    print(os.path.dirname(os.path.realpath(__file__)))
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = K.tf.Session(config=config)
    KTF.set_session(session)
    kwargs_dict = {'version_se': 0}
    model = get_symbol_binary(nn_input_shape=(112, 112, 3),
                              num_layers=34, embedding=512, **kwargs_dict)
    model.summary()
