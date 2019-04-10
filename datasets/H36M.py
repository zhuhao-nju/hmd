from __future__ import print_function 
import sys
sys.path.append("../src/")
import numpy as np
import PIL.Image
import cv2
import cPickle as pickle
import json
from tqdm import trange
from time import sleep
import renderer as rd
from scipy.io import loadmat
from mesh_edit import fast_deform_dja
from mesh_edit import fast_deform_dsa
from hmr_predictor import hmr_predictor
from hmr_predictor import proc_sil
from utility import make_trimesh
from utility import get_joint_move
from utility import get_achr_move
from utility import transform_h36m_joints
from utility import get_anchor_posi
from utility import take_notes
from utility import refine_sil


def proc_h36m(train_dir, test_dir, train_id, test_id, h36m_dir):
    
    sample_interval = 10
    
    faces = np.load("../predef/smpl_faces.npy")
    face_num = len(faces)
    
    with open ('../predef/mesh_joint_list.pkl', 'rb') as fp:
        mesh_joint = pickle.load(fp)
        
    hmr_pred = hmr_predictor()
    renderer = rd.SMPLRenderer(face_path = "../predef/smpl_faces.npy")
    
    # open available video list
    with open("./h36m_list.txt") as f:
        h36m_list = f.read().split("\r\n")
    vid_num = int(h36m_list[0])
    h36m_list = [[h36m_list[i*3+1], 
                  h36m_list[i*3+2],
                  h36m_list[i*3+3]] for i in range(vid_num)]
    
    # compute data number for training and testing
    train_num = int(vid_num * 0.8)
    test_num = vid_num - train_num

    count_all = 0.
    count_work = 0.

    # make test set
    tr = trange(test_num, desc='Bar desc', leave=True)
    for i in tr:
        tr.set_description("H36M - test part")
        tr.refresh() # to show immediately the update
        sleep(0.01)
        
        vid_idx = i + train_num
        
        # read video of image, silhouette and pose
        vid_img = cv2.VideoCapture(h36m_dir + h36m_list[vid_idx][0])
        vid_sil = cv2.VideoCapture(h36m_dir + h36m_list[vid_idx][1])
        pose_list = loadmat(h36m_dir + h36m_list[vid_idx][2])['pose']
        vid_len = min(int(vid_img.get(cv2.CAP_PROP_FRAME_COUNT)), 
                      int(vid_sil.get(cv2.CAP_PROP_FRAME_COUNT)),
                      len(pose_list))
        
        for frm_idx in range(0, vid_len, sample_interval):
            
            count_all += 1
            
            # read sil
            vid_sil.set(1, frm_idx)
            _,src_gt_sil = vid_sil.read()
            src_gt_sil[src_gt_sil<128] = 0
            src_gt_sil[src_gt_sil>=128] = 255
            src_gt_sil = refine_sil(src_gt_sil, 100)
            src_gt_sil = src_gt_sil[:, :, 0]
            
            # read ori img
            vid_img.set(1, frm_idx)
            _,ori_img = vid_img.read()
            # BGR to RGB
            ori_img = np.stack((ori_img[:,:,2], 
                                ori_img[:,:,1], 
                                ori_img[:,:,0]), axis = 2) 
            
            # hmr predict
            verts, cam, proc_para, std_img = hmr_pred.predict(ori_img, 
                                                              True, 
                                                              src_gt_sil)

            # unnormalize std_img
            src_img = ((std_img+1).astype(np.float)/2.0*255).astype(np.uint8)
            
            # save img
            img_file = "img/H36M_%04d%04d.png" % (vid_idx, frm_idx)
            PIL.Image.fromarray(src_img).save(test_dir + img_file)
    

            # process sil
            gt_sil = proc_sil(src_gt_sil, proc_para)

            # get proj sil
            proj_sil = renderer.silhouette(verts = verts,
                                           cam = cam,
                                           img_size = src_img.shape,
                                           norm = False)
        
            # make TriMesh
            mesh = make_trimesh(verts, faces, compute_vn = True)
            vert_norms = mesh.vertex_normals()
            
            h36m_joint = transform_h36m_joints(pose_list[frm_idx])
            # get joint move
            new_jv, _, joint_move, joint_posi = get_joint_move(verts, 
                                                   h36m_joint, 
                                                   proc_para,
                                                   mesh_joint)
            joint_move = joint_move.flatten()

            # joint deform
            fd_ja = fast_deform_dja(weight = 10.0)
            ja_verts = fd_ja.deform(np.asarray(verts), new_jv)


            # get achr move
            proj_sil_ja = renderer.silhouette(verts = ja_verts,
                                           norm = False)
            _, achr_verts, achr_move = get_achr_move(gt_sil, 
                                                     ja_verts, 
                                                     vert_norms,
                                                     proj_sil_ja)
            achr_posi = get_anchor_posi(achr_verts)
            
            # save sil
            compose_sil = np.stack((gt_sil, proj_sil, proj_sil_ja))
            compose_sil = np.moveaxis(compose_sil, 0, 2)
            compose_sil = PIL.Image.fromarray(compose_sil.astype(np.uint8))
            compose_sil.save(test_dir + "sil/%08d.png" % test_id)
            
            # save para
            proc_para['end_pt'] = proc_para['end_pt'].tolist()
            proc_para['start_pt'] = proc_para['start_pt'].tolist()
            para = {"verts": verts.tolist(),
                    "vert_norms": vert_norms.tolist(),
                    "proc_para": proc_para,
                    "joint_move": joint_move.tolist(),
                    "joint_posi": joint_posi.tolist(),
                    "achr_move": achr_move.tolist(),
                    "achr_posi": achr_posi.tolist(),
                    "img_file": img_file,
                   }
            with open(test_dir + "para/%08d.json" % test_id, 'wb') as fp:
                json.dump(para, fp)

            take_notes("H36M %04d%04d TEST %08d\n" % (vid_idx, frm_idx, 
                       test_id), "./data_log.txt")
            test_id += 1
            count_work += 1
            
    print("work ratio = %f, (%d / %d)" 
          % (count_work/count_all, count_work, count_all))
    return train_id, test_id
