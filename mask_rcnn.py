import os
import sys
import random
import math
import numpy as np
import skimage.io
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from nets.mrcnn import get_predict_model
from utils.config import Config
from utils.anchors import get_anchors
from utils.utils import mold_inputs, unmold_detections
from utils import visualize
import keras.backend as K


class MASK_RCNN(object):
    # logs/stage220210419_epoch020_loss0.183_val_loss0.953.h5
    _defaults = {
        # "model_path": 'logs/stage220210407_epoch100_loss0.510_val_loss1.498.h5',
        # "model_path": 'logs/stage220210528_epoch100_loss0.593_val_loss1.602.h5',
        # "model_path": 'logs/stage220210605_epoch200_loss0.354_val_loss1.504.h5',
        # "model_path": 'logs_0608/stage220210608_epoch100_loss0.580_val_loss1.628.h5',
        # "model_path": 'logs_0608/stage220210612_epoch149_loss0.422_val_loss1.477.h5',
        # "model_path": 'logs_0616/stage220210616_epoch050_loss0.883_val_loss1.797.h5',
        # "model_path": 'logs_0627/stage220210627_epoch050_loss0.691_val_loss1.256.h5',
        # "model_path": 'logs_0616/stage220210619_epoch100_loss0.524_val_loss1.500.h5',
        # "model_path": 'logs_0721/stage220210720_epoch007_loss7.009_val_loss7.208.h5',
        # "model_path": 'logs_0723/stage220210723_epoch058_loss5.696_val_loss7.207.h5',
        # "model_path": 'logs_0723/stage220210723_epoch060_loss5.662_val_loss7.249.h5',
        # "model_path": 'logs_0726/stage220210726_epoch015_loss3.617_val_loss4.271.h5',
        # "model_path": 'logs_0727/stage220210727_epoch090_loss0.526_val_loss1.542.h5',
        # "model_path": 'logs_0727/stage220210727_epoch082_loss0.565_val_loss1.528.h5',
        # "model_path": 'logs_0727/stage220210727_epoch100_loss0.493_val_loss1.581.h5',
        # "model_path": 'logs_0727/stage220210727_epoch062_loss0.736_val_loss1.657.h5',
        # "model_path": 'logs_0727/stage220210727_epoch069_loss0.657_val_loss1.592.h5',
        # "model_path": 'logs_0727/stage220210727_epoch096_loss0.503_val_loss1.504.h5',
        # "model_path": 'logs_0727/stage220210727_epoch033_loss1.621_val_loss2.386.h5',
        # "model_path": 'logs_0727/stage220210727_epoch045_loss1.090_val_loss1.971.h5',
        # "model_path": 'logs_0727/stage220210727_epoch066_loss0.688_val_loss1.648.h5',
        # "model_path": 'logs_0802/stage220210802_epoch013_loss1.043_val_loss1.378.h5',
        # "model_path": 'logs_0803/stage220210803_epoch013_loss0.925_val_loss1.292.h5',
        # "model_path": 'logs_0803/stage220210803_epoch020_loss0.807_val_loss1.303.h5',
        # "model_path": 'logs_0806/stage220210806_epoch009_loss0.996_val_loss1.273.h5',
        # "model_path": 'logs_0909/stage220210909_epoch004_loss0.960_val_loss1.143.h5',
        # "model_path": 'logs_0830/stage220210831_epoch004_loss0.956_val_loss1.129.h5',
        # "model_path": 'logs_0826/stage220210826_epoch007_loss0.845_val_loss1.121.h5',
        # "model_path": 'logs_0909/stage220210912_epoch003_loss0.877_val_loss1.054.h5',
        # "model_path": 'logs_0909/stage220210913_epoch003_loss0.831_val_loss1.060.h5',
        # "model_path": 'logs_0811/stage220210811_epoch009_loss1.069_val_loss1.242.h5',
        # "model_path": 'logs_0916/stage220210915_epoch008_loss0.820_val_loss1.123.h5',
        # "model_path": 'logs_0909/stage220210909_epoch004_loss0.960_val_loss1.143.h5',
        # "model_path": 'logs_0727/stage220210727_epoch100_loss0.493_val_loss1.581.h5',
        # "model_path": 'logs_0802/stage220210802_epoch015_loss0.983_val_loss1.442.h5',
        # "model_path": 'logs_0830/stage220210831_epoch004_loss0.956_val_loss1.129.h5',
        ## lll
        # "model_path": 'logs_0916/stage220210915_epoch008_loss0.820_val_loss1.123.h5',

        "model_path": 'logs_0727/stage220210727_epoch096_loss0.503_val_loss1.504.h5',
        # "model_path": 'logs_1007_2/stage220211008_epoch008_loss0.935_val_loss1.242.h5',
        # "model_path": 'logs_1008_1/stage220211008_epoch005_loss1.002_val_loss1.226.h5',
        "classes_path": 'model_data/shape_classes.txt',
        # "confidence": 0.90,
        "confidence": 0.85,

        # 使用coco数据集检测的时候，IMAGE_MIN_DIM=1024，IMAGE_MAX_DIM=1024, RPN_ANCHOR_SCALES=(32, 64, 128, 256, 512)
        # "RPN_ANCHOR_SCALES": (32, 64, 128, 256, 512),
        # "IMAGE_MIN_DIM": 1024,
        # "IMAGE_MAX_DIM": 1024,

        # 在使用自己的数据集进行训练的时候，如果显存不足要调小图片大小
        # 同时要调小anchors
        "IMAGE_MIN_DIM": 512,
        "IMAGE_MAX_DIM": 512,
        # "RPN_ANCHOR_SCALES": (16, 32, 64, 128, 256)
        "RPN_ANCHOR_SCALES": (8, 16, 32, 64, 128)
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Mask-Rcnn
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.sess = K.get_session()
        self.config = self._get_config()
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        class_names.insert(0, "BG")
        return class_names

    def _get_config(self):
        class InferenceConfig(Config):
            NUM_CLASSES = len(self.class_names)
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = self.confidence

            NAME = "shapes"
            RPN_ANCHOR_SCALES = self.RPN_ANCHOR_SCALES
            IMAGE_MIN_DIM = self.IMAGE_MIN_DIM
            IMAGE_MAX_DIM = self.IMAGE_MAX_DIM

        config = InferenceConfig()
        config.display()
        return config

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算总的种类
        self.num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        self.model = get_predict_model(self.config)
        self.model.load_weights(self.model_path, by_name=True)
     #   self.model.load_weights(self.model_path, by_name=True,
                     #      exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        image = [np.array(image)]
        molded_images, image_metas, windows = mold_inputs(self.config, image)

        image_shape = molded_images[0].shape
        anchors = get_anchors(self.config, image_shape)
        anchors = np.broadcast_to(anchors, (1,) + anchors.shape)

        detections, _, _, mrcnn_mask, _, _, _ = \
            self.model.predict([molded_images, image_metas, anchors], verbose=0)

        final_rois, final_class_ids, final_scores, final_masks = \
            unmold_detections(detections[0], mrcnn_mask[0],
                              image[0].shape, molded_images[0].shape,
                              windows[0])

        r = {
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        }



        # 想要保存处理后的图片请查询plt保存图片的方法。
        # drawed_image = visualize.display_instances(image[0], r['rois'], r['masks'], r['class_ids'],
        #                                           self.class_names, r['scores'])
        return r

    def detect_image1(self, image):
        image = [np.array(image)]
        molded_images, image_metas, windows = mold_inputs(self.config, image)

        image_shape = molded_images[0].shape
        anchors = get_anchors(self.config, image_shape)
        anchors = np.broadcast_to(anchors, (1,) + anchors.shape)

        detections, _, _, mrcnn_mask, _, _, _ = \
            self.model.predict([molded_images, image_metas, anchors], verbose=0)

        final_rois, final_class_ids, final_scores, final_masks = \
            unmold_detections(detections[0], mrcnn_mask[0],
                              image[0].shape, molded_images[0].shape,
                              windows[0])

        r = {
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        }

        #想要保存处理后的图片请查询plt保存图片的方法。
        drawed_image = visualize.display_instances(image[0], r['rois'], r['masks'], r['class_ids'],
                                                  self.class_names, r['scores'])
        return  drawed_image

    def close_session(self):
        self.sess.close()
