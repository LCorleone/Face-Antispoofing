import cv2
import numpy as np
from skimage import transform as trans

'''
def preprocess(img, bbox=None, landmark=None, image_size=None):
    M = None
    image_size = image_size
    if landmark is not None:
        assert len(image_size) == 2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        #M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

        if M is None:
            if bbox is None:  # use center crop
                det = np.zeros(4, dtype=np.int32)
                det[0] = int(img.shape[1] * 0.0625)
                det[1] = int(img.shape[0] * 0.0625)
                det[2] = img.shape[1] - det[0]
                det[3] = img.shape[0] - det[1]
            else:
                det = bbox
            margin = 48
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
            bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
            ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
            if len(image_size) > 0:
                ret = cv2.resize(ret, (image_size[1], image_size[0]))
            return ret
        else:  # do align using landmark
            assert len(image_size) == 2
            warped = cv2.warpAffine(
                img, M, (image_size[1], image_size[0]), borderValue=0.0)
            return warped
    else:

        # det = np.zeros(4, dtype=np.int32)
        # det[0] = int(img.shape[1]*0.0625)
        # det[1] = int(img.shape[0]*0.0625)
        # det[2] = img.shape[1] - det[0]
        # det[3] = img.shape[0] - det[1]

        # margin = 48
        # bb = np.zeros(4, dtype=np.int32)
        # bb[0] = np.maximum(det[0]-margin/2, 0)
        # bb[1] = np.maximum(det[1]-margin/2, 0)
        # bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        # bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = int(img.shape[1] / 2) - int(image_size[1] / 2)
        bb[2] = bb[0] + image_size[1]
        bb[1] = int(img.shape[0] / 2) - int(image_size[0] / 2)
        bb[3] = bb[0] + image_size[0]

        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
'''


def preprocess(img, bbox=None, landmark=None, image_size=None):
    M = None
    image_size = image_size
    if landmark is not None:
        assert len(image_size) == 2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        if image_size[0] == 64:
            src = np.array([
                [19.2946, 30.6963],
                [45.5318, 30.5014],
                [32.0252, 43.7366],
                [23.5493, 52.3655],
                [41.7299, 52.2041]], dtype=np.float32)
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]

    if M is None:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = 44
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2
        warped = cv2.warpAffine(
            img, M, (image_size[1], image_size[0]), borderValue=0.0)
        return warped
