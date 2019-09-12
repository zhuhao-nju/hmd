from torch.utils.data.dataset import Dataset
import numpy as np
import PIL.Image
import random
import json
import pickle
import torchvision.transforms as transforms
from utility import center_crop

import configparser
conf = configparser.ConfigParser()
conf.read(u'../conf.ini', encoding='utf8')
tgt_path = conf.get('DATA', 'tgt_path')

#==============================================================================
# data loader for joint train/test
#==============================================================================
class dataloader_joint(Dataset):
    def __init__(self, 
                 sil_ver = False,
                 train = True,
                 transform = transforms.Compose([transforms.ToTensor()]),
                 manual_seed = 1234,
                 shuffle = True,
                 get_all = False,
                ):
        if train is True:
            self.dataset_dir = tgt_path + "train/"
            self.num = 24149*10 #13938, non dr augment version
        else:
            self.dataset_dir = tgt_path + "test/"
            self.num = 4625*10

        # transfer arg
        self.sil_ver = sil_ver
        self.transform = transform
        self.get_all = get_all
        
        # make random seed
        if manual_seed is None:
            manual_seed = random.randint(1, 10000)
        random.seed(manual_seed)
        
        # make data_id list and shuffle
        self.id_list = list(range(self.num))
        if shuffle is True:
            random.shuffle(self.id_list)

    def __len__(self):
        return self.num
        
    def __getitem__(self, index):
        crt_id = self.id_list[index]
        img_id = int(np.floor(crt_id / 10))
        joint_id = crt_id % 10
        
        # get sil
        all_sil = np.array(PIL.Image.open(self.dataset_dir + 
                                      "sil/%08d.png" % img_id))
        all_sil[all_sil<128] = 0
        all_sil[all_sil>=128] = 255
        
        # get parameters
        with open (self.dataset_dir + "/para/%08d.json" % img_id, 'rb') as fp:
            para_dic = json.load(fp)
        joint_move = para_dic["joint_move"]
        joint_posi = para_dic["joint_posi"]
        
        # make target para
        tgt_para = np.array(joint_move[(joint_id*2):(joint_id*2+2)])
        
        if self.sil_ver is False:
            # make input array for image version
            img_file = para_dic["img_file"]
            src_img = np.array(PIL.Image.open(self.dataset_dir + img_file))
            src_sil = np.expand_dims(all_sil[:,:,1], 2)
            crop_sil = center_crop(src_sil, joint_posi[joint_id], 64)
            crop_img = center_crop(src_img, joint_posi[joint_id], 64)
            crop_img = crop_img.astype(np.int)
            crop_img = crop_img - crop_img[31, 31, :]
            crop_img = np.absolute(crop_img)

            src_in = np.concatenate((crop_sil, crop_img), axis = 2)
        else:
            # make input array for silhouette version
            src_sil = all_sil[:,:,:2]
            src_in = center_crop(src_sil, joint_posi[joint_id], 64)
            
        # transform as torch tensor
        src_in = PIL.Image.fromarray(src_in.astype(np.uint8))
        if self.transform != None:
            src_in = self.transform(src_in)
        
        if self.get_all is True and self.sil_ver is False:
            # get verts and vert_norms
            verts = np.array(para_dic["verts"])
            vert_norms = np.array(para_dic["vert_norms"])
            proc_para = para_dic["proc_para"]
            return (src_in, tgt_para, src_img, verts, 
                    vert_norms, proc_para, all_sil)
        else:
            return (src_in, tgt_para)
        

#==============================================================================
# data loader for anchor train/test
#==============================================================================
class dataloader_anchor(Dataset):
    def __init__(self, 
                 sil_ver = False,
                 train = True,
                 transform = transforms.Compose([transforms.ToTensor()]),
                 manual_seed = 1234,
                 shuffle = True,
                 get_all = False,
                ):
        if train is True:
            self.dataset_dir = tgt_path + "train/"
            self.num = 24149*200 #13938, non dr augment version
        else:
            self.dataset_dir = tgt_path + "test/"
            self.num = 4625*200

        # transfer arg
        self.sil_ver = sil_ver
        self.transform = transform
        self.get_all = get_all
        
        # make random seed
        if manual_seed is None:
            manual_seed = random.randint(1, 10000)
        random.seed(manual_seed)
        
        # make data_id list and shuffle
        self.id_list = list(range(self.num))
        if shuffle is True:
            random.shuffle(self.id_list)

    def __len__(self):
        return self.num
        
    def __getitem__(self, index):
        crt_id = self.id_list[index]
        img_id = int(np.floor(crt_id / 200))
        achr_id = crt_id % 200
        
        # get sil
        all_sil = np.array(PIL.Image.open(self.dataset_dir + 
                                      "sil/%08d.png" % img_id))
        all_sil[all_sil<128] = 0
        all_sil[all_sil>=128] = 255
        
        # get parameters
        with open (self.dataset_dir + "/para/%08d.json" % img_id, 'rb') as fp:
            para_dic = json.load(fp)
        achr_move = np.array(para_dic["achr_move"])
        achr_posi = para_dic["achr_posi"]
        
        # get source image
        img_file = para_dic["img_file"]
        src_img = np.array(PIL.Image.open(self.dataset_dir + img_file))
        
        tgt_para = achr_move[achr_id]
        tgt_para = np.expand_dims(tgt_para, 0)
        
        
        if self.sil_ver is False:
            src_sil = np.expand_dims(all_sil[:,:,2], 2)
            crop_sil = center_crop(src_sil, achr_posi[achr_id], 32)
            crop_img = center_crop(src_img, achr_posi[achr_id], 32)
            crop_img = crop_img.astype(np.int)
            crop_img = crop_img - crop_img[15, 15, :]
            crop_img = np.absolute(crop_img)
            src_in = np.concatenate((crop_sil, crop_img), axis = 2)
        else:
            # make input array for silhouette version
            src_sil = np.stack((all_sil[:,:,0], all_sil[:,:,2]), axis = -1)
            src_in = center_crop(src_sil, achr_posi[achr_id], 32)
            
        # transform as torch tensor
        src_in = PIL.Image.fromarray(src_in.astype(np.uint8))
        if self.transform != None:
            src_in = self.transform(src_in)
            
        if self.get_all is True and self.sil_ver is False:
            # get verts and vert_norms
            verts = np.array(para_dic["verts"])
            vert_norms = np.array(para_dic["vert_norms"])
            proc_para = para_dic["proc_para"]
            return (src_in, tgt_para, src_img, verts, 
                    vert_norms, proc_para, all_sil)
        else:
            return (src_in, tgt_para)
        
        
#==============================================================================
# data loader for shading training/testing
#==============================================================================    
class dataloader_shading(Dataset):
    def __init__(self, 
                 train = True,
                 manual_seed = 1234,
                 shuffle = True,
                ):
        self.dataset_dir = "/media/hao/DATA/ShadingData/data1006/"
        self.num = 2273
            
        # make random seed
        if manual_seed is None:
            manual_seed = random.randint(1, 10000)
        random.seed(manual_seed)
        
        # make data_id list and shuffle
        self.id_list = list(range(self.num))
        if shuffle is True:
            random.shuffle(self.id_list)

    def __len__(self):
        return self.num
        
    def __getitem__(self, index):
        tuple_id = self.id_list[index]
        
        # get source image
        src_img = np.array(PIL.Image.open(self.dataset_dir + 
                                          "inputC_%04d.jpg" % tuple_id))
        src_img = np.rollaxis(src_img, 2, 0) / 256.0
        
        # get gt depth
        f_dgt = open(self.dataset_dir + 'gtD_%04d.bin' % tuple_id, "rb")
        depth_gt = np.resize(np.fromfile(f_dgt, dtype=np.float32), 
                             (448, 448)).transpose()
        
        # get smooth depth
        f_dsm = open(self.dataset_dir + 'smoothD_%04d.bin' % tuple_id, "rb")
        depth_sm = np.resize(np.fromfile(f_dsm, dtype=np.float32), 
                             (448, 448)).transpose()
        
        # compute depth difference
        depth_diff = depth_gt - depth_sm
        depth_diff = depth_diff * 10
        depth_diff = np.expand_dims(depth_diff, 0)
        
        # get mask
        mask = np.zeros(depth_diff.shape)
        mask[depth_diff!=0] = 1
        
        return (src_img, depth_diff, mask)
                
#==============================================================================
# data loader for efficient predicting test
#==============================================================================
class dataloader_pred(Dataset):
    def __init__(self, 
                 train = True,
                 manual_seed = 1234,
                 shuffle = True,
                 dataset_path = None,
                ):
        if dataset_path is None:
            dataset_path = tgt_path
        if train is True:
            self.dataset_dir = dataset_path+"train/"
            self.num = 24149 #13938, non dr augment version
        else:
            self.dataset_dir = dataset_path+"test/"
            self.num = 4625

        # make random seed
        if manual_seed is None:
            manual_seed = random.randint(1, 10000)
        random.seed(manual_seed)
        
        # make data_id list and shuffle
        self.id_list = list(range(self.num))
        if shuffle is True:
            random.shuffle(self.id_list)

    def __len__(self):
        return self.num
        
    def __getitem__(self, index):
        img_id = self.id_list[index]
        
        # get sil
        all_sil = np.array(PIL.Image.open(self.dataset_dir + 
                                      "sil/%08d.png" % img_id))
        all_sil[all_sil<128] = 0
        all_sil[all_sil>=128] = 1
        
        # get parameters
        with open (self.dataset_dir + "/para/%08d.json" % img_id, 'rb') as fp:
            para_dic = json.load(fp)
        
        # get src_img and pre-processing parameters
        img_file = para_dic["img_file"]
        src_img = np.array(PIL.Image.open(self.dataset_dir + img_file))
        proc_para = para_dic["proc_para"]
        
        # get verts and vert_norms
        verts = np.array(para_dic["verts"])
        vert_norms = np.array(para_dic["vert_norms"])
        
        # get joint move and position
        joint_move = np.array(para_dic["joint_move"])
        joint_posi = para_dic["joint_posi"]
        
        # get anchor move and position
        achr_move = np.array(para_dic["achr_move"])
        achr_posi = para_dic["achr_posi"]
        
        # make source for joint net
        sil_j = np.expand_dims(all_sil[:,:,1], 2)
        src_j = np.zeros((10, 4, 64, 64))
        for i in range(len(joint_posi)):
            crop_sil = center_crop(sil_j, joint_posi[i], 64)
            crop_img = center_crop(src_img, joint_posi[i], 64)
            crop_img = crop_img.astype(np.float)
            crop_img = crop_img - crop_img[31, 31, :]
            crop_img = np.absolute(crop_img)
            crop_img = crop_img/255.0
            src_j[i,0,:,:] = np.rollaxis(crop_sil, 2, 0)
            src_j[i,1:4,:,:] = np.rollaxis(crop_img, 2, 0)
        
        # make source for anchor net
        src_a = None
        # commentted because prediction didn't require this
        '''
        sil_a = np.stack((all_sil[:,:,0], all_sil[:,:,2]), axis = -1)
        src_a = np.zeros((200, 2, 32, 32))
        for i in range(len(achr_posi)):
            crop_sil = center_crop(sil_a, achr_posi[i], 32)
            src_a[i,:,:,:] = np.rollaxis(crop_sil, 2, 0)
        '''
        return (src_j, src_a, src_img, joint_move, achr_move, verts, 
                vert_norms, proc_para, all_sil, joint_posi, achr_posi)
    
                
#==============================================================================
# data loader for efficient predicting test, sil version
#==============================================================================
class dataloader_sil_pred(Dataset):
    def __init__(self, 
                 train = True,
                 manual_seed = 1234,
                 shuffle = True,
                 dataset_path = None,
                ):
        if dataset_path is None:
            dataset_path = tgt_path
        if train is True:
            self.dataset_dir = dataset_path + "train/"
            self.num = 24149 #13938, non dr augment version
        else:
            self.dataset_dir = dataset_path + "test/"
            self.num = 4625

        # make random seed
        if manual_seed is None:
            manual_seed = random.randint(1, 10000)
        random.seed(manual_seed)
        
        # make data_id list and shuffle
        self.id_list = list(range(self.num))
        if shuffle is True:
            random.shuffle(self.id_list)

    def __len__(self):
        return self.num
        
    def __getitem__(self, index):
        img_id = self.id_list[index]
        
        # get sil
        all_sil = np.array(PIL.Image.open(self.dataset_dir + 
                                      "sil/%08d.png" % img_id))
        all_sil[all_sil<128] = 0
        all_sil[all_sil>=128] = 1
        
        # get parameters
        with open (self.dataset_dir + "/para/%08d.json" % img_id, 'rb') as fp:
            para_dic = json.load(fp)
        
        # get src_img and pre-processing parameters
        img_file = para_dic["img_file"]
        src_img = np.array(PIL.Image.open(self.dataset_dir + img_file))
        proc_para = para_dic["proc_para"]
        
        # get verts and vert_norms
        verts = np.array(para_dic["verts"])
        vert_norms = np.array(para_dic["vert_norms"])
        
        # get joint move and position
        joint_move = np.array(para_dic["joint_move"])
        joint_posi = para_dic["joint_posi"]
        
        # get anchor move and position
        achr_move = np.array(para_dic["achr_move"])
        achr_posi = para_dic["achr_posi"]
        
        # make source for joint net
        sil_j = all_sil[:,:,:2]
        src_j = np.zeros((10, 2, 64, 64))
        for i in range(len(joint_posi)):
            crop_sil = center_crop(sil_j, joint_posi[i], 64)
            src_j[i,:,:,:] = np.rollaxis(crop_sil, 2, 0)
        
        # make source for anchor net
        src_a = None
        # commentted because prediction didn't require this
        '''
        sil_a = np.stack((all_sil[:,:,0], all_sil[:,:,2]), axis = -1)
        src_a = np.zeros((200, 2, 32, 32))
        for i in range(len(achr_posi)):
            crop_sil = center_crop(sil_a, achr_posi[i], 32)
            src_a[i,:,:,:] = np.rollaxis(crop_sil, 2, 0)
        '''
        return (src_j, src_a, src_img, joint_move, achr_move, verts, 
                vert_norms, proc_para, all_sil, joint_posi, achr_posi)
    
    
#==============================================================================
# data loader for efficient demo (no ground-truth reading, only test)
#==============================================================================
class dataloader_demo(Dataset):
    def __init__(self, dataset_path):
        self.dataset_dir = dataset_path+"test/"
        self.num = 4625
        self.id_list = list(range(self.num))

    def __len__(self):
        return self.num
        
    def __getitem__(self, index):
        img_id = self.id_list[index]
        
        # get parameters
        with open (self.dataset_dir + "/para/%08d.json" % img_id, 'rb') as fp:
            para_dic = json.load(fp)
        
        # get src_img and pre-processing parameters
        img_file = para_dic["img_file"]
        src_img = np.array(PIL.Image.open(self.dataset_dir + img_file))
        proc_para = para_dic["proc_para"]
        
        # get verts and vert_norms
        verts = np.array(para_dic["verts"])
        vert_norms = np.array(para_dic["vert_norms"])
        
        return (src_img, verts, vert_norms)
    
                
