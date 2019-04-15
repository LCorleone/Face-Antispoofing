import os
import cv2
import pdb
import tensorflow as tf
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
import sys
import pickle
import pdb
import numpy as np
sys.path.append('./network')
from resnet100 import get_symbol_binary
from func import *
sys.path.append('./mtcnn')
import face_preprocess
import detect_face


def draw_box(image, box, color, thickness=4):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]),
                  color, thickness, cv2.LINE_AA)


class AntiSpoofing(object):
    """docstring for Video_Face_Recognition"""

    def __init__(self, save_path, video_out_name, model_params, MTCNN_params, gpu_id, threshold, frame_period):
        super(AntiSpoofing, self).__init__()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = K.tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = K.tf.Session(config=config)
        KTF.set_session(session)
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.video_out_name = video_out_name
        self.model_params = model_params
        self.MTCNN_params = MTCNN_params
        self.threshold = threshold
        self.result_queue = []
        self.result_queue_length = frame_period

        print('initilize the face detection network ...\n')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(allow_growth=True)
            # sess = tf.Session()
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(
                    sess, None)

        print('initilize the AntiSpoofing network ...\n')
        kwargs_dict = {'version_se': self.model_params['version_se']}
        self.antispoofing_model = get_symbol_binary(nn_input_shape=(self.model_params['image_size'][0], self.model_params['image_size'][1], 3),
                                                    num_layers=self.model_params['num_layers'],
                                                    embedding=self.model_params['embedding'], **kwargs_dict)
        self.antispoofing_model.load_weights(self.model_params['model_path'])

    def despoof(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bounding_boxes, points = detect_face.detect_face(
            img, self.MTCNN_params['minsize'], self.pnet, self.rnet, self.onet, self.MTCNN_params['threshold'], self.MTCNN_params['factor'])
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces >= 1:
            for index, bbox in enumerate(bounding_boxes):
                score = bbox[-1]
                bbox = bbox[0: 4]
                landmark = points[:, index].reshape((2, 5)).T
            if score > 0.8:
                # print(bbox[3] - bbox[1], bbox[2] - bbox[0])
                warped = face_preprocess.preprocess(
                    img, bbox=bbox, landmark=landmark, image_size=(112, 112))
                warped__ = preprocess_input(warped)
                to_test = np.expand_dims(warped__, 0)
                prediction = self.antispoofing_model.predict(to_test)
                # 0.001
                if prediction <= self.threshold:
                    # draw_box(img_bgr, [bbox[0], bbox[1], bbox[2],
                    #                    bbox[3]], color=[127, 255, 0])
                    # video_writer.write(img_bgr.astype(np.uint8))
                    self.pop_queue(1)
                else:
                    # draw_box(img_bgr, [bbox[0], bbox[1], bbox[2],
                    #                    bbox[3]], color=[0, 0, 255])
                    self.pop_queue(0)
            else:
                self.pop_queue(0)
        else:
            self.pop_queue(0)

        if self.get_result():
            draw_box(img_bgr, [bbox[0], bbox[1], bbox[2],
                               bbox[3]], color=[127, 255, 0])
        else:
            draw_box(img_bgr, [bbox[0], bbox[1], bbox[2],
                               bbox[3]], color=[0, 0, 255])
        return img_bgr

    def pop_queue(self, k):
        if len(self.result_queue) <= self.result_queue_length:
            self.result_queue.append(k)
        else:
            self.result_queue.pop(0)
            self.result_queue.append(k)

    def get_result(self):
        if np.sum(self.result_queue) > self.result_queue_length / 2:
            return True
        else:
            return False


if __name__ == '__main__':

    MTCNN_params = {'minsize': 20, 'threshold': [
        0.6, 0.7, 0.9], 'factor': 0.7}
    model_params = {'version_se': 0, 'image_size': (112, 112), 'num_layers': 100, 'embedding': 512,
                    'model_path': './antispoofing_model_resnet100/model_ex-002_loss-0.007775.h5'}

    save_path = './result_videos'
    video_name = 'resnet100_G_1_iphone'

    anti_obj = AntiSpoofing(save_path=save_path, video_out_name=video_name,
                            model_params=model_params, MTCNN_params=MTCNN_params, gpu_id=2, threshold=0.001, frame_period=30)

    video_path = './test_videos/G_1_iphone.mp4'
    vc = cv2.VideoCapture(video_path)
    video_writer = cv2.VideoWriter(os.path.join(save_path, 'output_' + video_name) + '.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (round(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), round(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    frame_count = 0
    print('video_path: {}'.format(video_path))
    while True:
        rval, img_bgr = vc.read()
        if rval is False:
            print(rval)
            break
        frame_count += 1
        img2save = anti_obj.despoof(img_bgr)
        if frame_count >= 30:
            video_writer.write(img2save.astype(np.uint8))
        result = anti_obj.get_result()
        print(frame_count, result)

