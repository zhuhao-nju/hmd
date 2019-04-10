from __future__ import print_function 
import numpy as np
import sys
import tensorflow as tf
from utility import shift_verts
from utility import resize_img

# cofigure hmr path
import configparser
conf = configparser.ConfigParser()
conf.read(u'../conf.ini', encoding='utf8')
hmr_path = conf.get('HMR', 'hmr_path')
sys.path.append(hmr_path)
from absl import flags
import src.config
from src.RunModel import RunModel

def proc_sil(src_sil, proc_para):
    scale = proc_para["scale"],
    start_pt = proc_para["start_pt"]
    end_pt = proc_para["end_pt"]
    img_size = proc_para["img_size"]
    
    sil_scaled, _ = resize_img(src_sil, scale)
    
    margin = int(img_size / 2)
    sil_pad = np.pad(sil_scaled.tolist(), 
                     ((margin, ), (margin, )), 
                     mode='constant') # use 0 to fill the padding area of sil
    sil_pad = np.asarray(sil_pad).astype(np.uint8)
    std_sil = sil_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]]
    return std_sil


class hmr_predictor():
    def __init__(self):
        self.config = flags.FLAGS
        self.config([1, "main"])
        self.config.load_path = src.config.PRETRAINED_MODEL
        self.config.batch_size = 1
        self.sess = tf.Session()
        self.model = RunModel(self.config, sess=self.sess)
    
    def predict(self, img, sil_bbox = False, sil = None):
        if sil_bbox is True:
            std_img, proc_para = preproc_img(img, True, sil)
        else:
            std_img, proc_para = preproc_img(img)
        std_img = np.expand_dims(std_img, 0)# Add batch dimension
        _, verts, ori_cam, _ = self.model.predict(std_img)
        shf_vert = shift_verts(proc_para, verts[0], ori_cam[0])
        cam = np.array([500, 112.0, 112.0])
        return shf_vert, cam, proc_para, std_img[0]
    
    def close(self):
        self.sess.close()
