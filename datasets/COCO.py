from __future__ import print_function 
import numpy as np
import PIL.Image
import skimage.io as io
import pickle
import json
from tqdm import trange
from time import sleep
import os.path
import sys
sys.path.append("../src/")
import renderer as rd
from mesh_edit import fast_deform_dja
from mesh_edit import fast_deform_dsa
from hmr_predictor import hmr_predictor
from hmr_predictor import proc_sil
from utility import make_trimesh
from utility import get_joint_move
from utility import get_achr_move
from utility import points2sil
from utility import get_anchor_posi
from data_filter import coco_filter
from utility import take_notes
from utility import transform_coco_joints

# cofigure coco_api_path
import configparser
conf = configparser.ConfigParser()
conf.read(u'../conf.ini', encoding='utf8')
coco_api_path = conf.get('DATA', 'coco_api_path')
sys.path.append(coco_api_path)
from pycocotools.coco import COCO


# all 10000 are train set
def proc_coco(train_dir, test_dir, train_id, test_id, coco_dataset):
    
    # read dataset
    coco=COCO(coco_dataset)
    tupleIds = coco.getImgIds(catIds=1) # id = 1 means person
    
    faces = np.load("../predef/smpl_faces.npy")
    face_num = len(faces)
    
    hmr_pred = hmr_predictor()
    renderer = rd.SMPLRenderer(face_path = 
                               "../predef/smpl_faces.npy")
    
    with open ('../predef/mesh_joint_list.pkl', 'rb') as fp:
        mesh_joint = pickle.load(fp)
    
    count_all = 0.
    count_work = 0.
    
    total_num = len(tupleIds)
    train_num = int(np.floor(total_num*0.8))
    
    # make train set
    tr = trange(train_num, desc='Bar desc', leave=True)
    for i in tr:
        tr.set_description("COCO - train part")
        tr.refresh() # to show immediately the update
        sleep(0.01)
        
        count_all += 1
        
        # get tuple
        one_tuple = coco.loadImgs(tupleIds[i])[0]
        img_size = (one_tuple['height'], one_tuple['width'])
        crt_id = one_tuple['id']
        
        # get anns
        annIds = coco.getAnnIds(imgIds=one_tuple['id'], catIds=1, iscrowd=None)
        anns = coco.loadAnns(annIds)
        
        # RULE 1: objects < 5
        if len(anns)>4:
            #print("filter out by too many objects")
            take_notes("COCO %05d BAN -1\n" % (i), "./data_log.txt")
            continue
        
        for j in range(len(anns)):
            
            # get sil points
            seg_points = anns[j]['segmentation'][0]
            
            # RULE 2: seg_points number >= 80
            if len(seg_points)<80:
                take_notes("COCO %05d%03d BAN -1\n" % (i, j), "./data_log.txt")
                #print("filter out by too few seg_points number")
                continue

            # get key points
            key_points = anns[j]['keypoints']
            key_points = np.resize(key_points,(17,3))
            
            # draw sil
            sil = points2sil(seg_points, img_size)

            result = coco_filter(key_points, sil)
            
            if result is False:
                take_notes("COCO %05d BAN -1\n" % (i), "./data_log.txt")
                continue
            # Here we finally decide to use it
            if result is True:
                # read image
                ori_img = io.imread(one_tuple['coco_url'])
            
                # read sil
                src_gt_sil = sil
                
                # hmr predict
                verts, cam, proc_para, std_img = hmr_pred.predict(ori_img, 
                                                                  True, 
                                                                  src_gt_sil)
                
                # unnormalize std_img
                src_img = ((std_img+1)/2.0*255).astype(np.uint8)
        
                # save img
                img_file = "img/COCO_%08d%02d.png" % (crt_id, j)
                PIL.Image.fromarray(src_img).save(train_dir + img_file)
                
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

                # get joint move
                coco_joints_t = transform_coco_joints(key_points)
                new_jv, _, joint_move, joint_posi = get_joint_move(verts, 
                                                       coco_joints_t,
                                                       proc_para,
                                                       mesh_joint,
                                                       unseen_mode = True,
                                                      )
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
                compose_sil.save(train_dir + "sil/%08d.png" % train_id)

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
                with open(train_dir + "para/%08d.json" % train_id, 'wb') as fp:
                    json.dump(para, fp)
                
                take_notes("COCO %05d%03d TRAIN %08d\n" % (i, j, train_id), 
                           "./data_log.txt")
                train_id += 1
                count_work += 1
    
    
    # make test set
    test_num = total_num - train_num
    tr = trange(test_num, desc='Bar desc', leave=True)
    for i in tr:
        tr.set_description("COCO - test part")
        tr.refresh() # to show immediately the update
        sleep(0.01)
        
        count_all += 1
        
        # get tuple
        one_tuple = coco.loadImgs(tupleIds[i+train_num])[0]
        img_size = (one_tuple['height'], one_tuple['width'])
        crt_id = one_tuple['id']
        
        # get anns
        annIds = coco.getAnnIds(imgIds=one_tuple['id'], catIds=1, iscrowd=None)
        anns = coco.loadAnns(annIds)
        
        # RULE 1: objects < 4
        if len(anns)>3:
            #print("filter out by too many objects")
            take_notes("COCO %05d BAN -1\n" % (i+train_num), "./data_log.txt")
            continue
        
        for j in range(len(anns)):
            
            # get sil points
            seg_points = anns[j]['segmentation'][0]

            # RULE 2: seg_points number >= 100
            if len(seg_points)<100:
                take_notes("COCO %05d%03d BAN -1\n" % (i+train_num, j), "./data_log.txt")
                #print("filter out by too few seg_points number")
                continue

            # get key points
            key_points = anns[j]['keypoints']
            key_points = np.resize(key_points,(17,3))
            
            # draw sil
            sil = points2sil(seg_points, img_size)

            result = coco_filter(key_points, sil)
            
            if result is False:
                take_notes("COCO %05d BAN -1\n" % (i+train_num), "./data_log.txt")
                continue
            # Here we finally decide to use it
            if result is True:
                # read image
                ori_img = io.imread(one_tuple['coco_url'])
            
                # read sil
                src_gt_sil = sil
                
                # hmr predict
                verts, cam, proc_para, std_img = hmr_pred.predict(ori_img, 
                                                                  True, 
                                                                  src_gt_sil)
                
                # unnormalize std_img
                src_img = ((std_img+1)/2.0*255).astype(np.uint8)
        
                # save img
                img_file = "img/COCO_%08d%02d.png" % (crt_id, j)
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

                # get joint move
                coco_joints_t = transform_coco_joints(key_points)
                new_jv, _, joint_move, joint_posi = get_joint_move(verts, 
                                                       coco_joints_t,
                                                       proc_para,
                                                       mesh_joint,
                                                       unseen_mode = True,
                                                      )
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
                
                take_notes("COCO %05d%03d TEST %08d\n" % (i+train_num, j, test_id), 
                           "./data_log.txt")
                test_id += 1
                count_work += 1
    
    
    print("work ratio = %f, (%d / %d)" 
          % (count_work/count_all, count_work, count_all))
    return train_id, test_id

