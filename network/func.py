from sklearn.utils import shuffle
import math
import os
import sys
import numpy as np
from keras import backend as K
from PIL import Image as pil_image
from keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize
import cv2


def load_img(path, grayscale=False, target_size=None, mode='rgb'):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
    """

    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if mode == 'hsv':
        img = img.convert('HSV')
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img


def img_to_array(img, data_format=None, preprocess_function='normal'):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)

    if preprocess_function == 'normal':
        x = preprocess_input(x)
    return x


def gen_img_path_label(img_folder):
    img_path = []
    label = []
    for person_id in os.listdir(img_folder):
        if person_id == 'genuine':
            train_id = 0
        elif person_id == 'forgery':
            train_id = 1
        for img in os.listdir(os.path.join(img_folder, person_id)):
            img_path.append(os.path.join(img_folder, person_id, img))
            label.append(train_id)
    return img_path, label


def img_generator(img_path_list, label, image_shape=(112, 112, 3), batchsize=32, if_shuffle=True):
    if if_shuffle:
        img_path_list, label = shuffle(img_path_list, label)

    print('total img is {}, and from total subject {}'.format(
        len(label), len(set(label))))
    # iter
    count = 0
    while 1:
        batch_x = np.zeros((batchsize,) + image_shape, dtype=K.floatx())
        batch_y = np.zeros((batchsize,), dtype=np.int32)
        i_index = 0
        for index in range(count * batchsize, (count + 1) * batchsize):
            img = load_img(img_path_list[index],
                           grayscale=False,
                           target_size=(image_shape[0], image_shape[1]))
            x = img_to_array(img)
            batch_x[i_index] = x
            batch_y[i_index] = label[index]
            i_index += 1

        if count < math.floor(len(img_path_list) / float(batchsize)) - 1:
            count += 1
        else:
            if if_shuffle:
                img_path_list, label = shuffle(img_path_list, label)
            count = 0
        # print('here')
        # print('size of batch x: {}'.format(sys.getsizeof(batch_x)))
        yield (batch_x, batch_y)
        # return (batch_x, batch_y)


def preprocess_input(x):
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(K.floatx(), copy=False)
    x -= 127.5
    x /= 128.
    return x


def flip_axis(x, axis):
    x = x.swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


class Image_Generator_sequence(Sequence):

    # xset is the list of the image path, yset is the list of labels corresponding
    def __init__(self, x_set, y_set, batch_size, random_flip, mode='rgb', img_size=(112, 112)):
        # x_set, y_set = shuffle(x_set, y_set)
        self.x, self.y = np.array(x_set), np.array(y_set)
        self.batch_size = batch_size
        self.indices = np.arange(len(x_set))
        self.random_flip = random_flip
        self.img_size = img_size
        self.mode = mode
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x_path = self.x[inds]
        batch_y_path = self.y[inds]

        batch_x = np.zeros(
            (self.batch_size,) + (self.img_size[0], self.img_size[1], 3), dtype=K.floatx())
        batch_y = np.zeros((self.batch_size,), dtype=np.int32)
        i_index = 0
        for index, file_name in enumerate(batch_x_path):
            if self.mode == 'rgb':
                img = load_img(file_name,
                               grayscale=False,
                               target_size=self.img_size)
            elif self.mode == 'hsv':
                img = load_img(file_name,
                               grayscale=False,
                               target_size=self.img_size, mode='hsv')
            x = img_to_array(img)
            if self.random_flip:
                # x = tf.image.random_flip_left_right(x)
                if np.random.random() < 0.5:
                    x = flip_axis(x, 1)
            batch_x[i_index] = x
            batch_y[i_index] = batch_y_path[index]
            i_index += 1
        # print('here')
        return (batch_x, batch_y)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


class Image_Generator_sequence_dual(Sequence):

    # xset is the list of the image path, yset is the list of labels corresponding
    def __init__(self, x_set, y_set, batch_size, random_flip, img_size=(112, 112)):
        # x_set, y_set = shuffle(x_set, y_set)
        self.x, self.y = np.array(x_set), np.array(y_set)
        self.batch_size = batch_size
        self.indices = np.arange(len(x_set))
        self.random_flip = random_flip
        self.img_size = img_size
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x_path = self.x[inds]
        batch_y_path = self.y[inds]

        batch_x = np.zeros(
            (self.batch_size,) + (self.img_size[0], self.img_size[1], 6), dtype=K.floatx())

        batch_y = np.zeros((self.batch_size,), dtype=np.int32)
        i_index = 0
        for index, file_name in enumerate(batch_x_path):
            img_bgr = cv2.imread(file_name)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV_FULL)
            x = np.concatenate((img_rgb, img_hsv), axis=-1)
            x = preprocess_input(x)
            batch_x[i_index] = x
            batch_y[i_index] = batch_y_path[index]
            i_index += 1
        # print('here')
        # X = [batch_x, batch_x_hsv]
        return (batch_x, batch_y)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
