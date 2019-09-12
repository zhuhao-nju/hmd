import numpy as np
import PIL.Image
import cv2, pickle, sys, os
from tqdm import trange
sys.path.append("../src/")
from predictor import joint_predictor
from predictor import anchor_predictor
from data_loader import dataloader_pred
from mesh_edit import fast_deform_dja
from mesh_edit import fast_deform_dsa
from renderer import SMPLRenderer
from utility import sil_iou
from utility import show_img_arr
from utility import center_crop
from utility import get_anchor_posi
from utility import get_joint_posi

pdt_j = joint_predictor("../demo/pretrained_model/pretrained_joint.pth")
pdt_a = anchor_predictor("../demo/pretrained_model/pretrained_anchor.pth")

dataset = dataloader_pred(dataset_path = "./eval_data/wild_set/",
                          train = False, shuffle = False)
my_renderer = SMPLRenderer(face_path="../predef/smpl_faces.npy")

sta_num = 2699

# make verts for joint deform
with open ('../predef/mesh_joint_list.pkl', 'rb') as fp:
    item_dic = pickle.load(fp)
point_list = item_dic["point_list"]
index_map = item_dic["index_map"]

# make verts for anchor deform
with open ('../predef/dsa_achr.pkl', 'rb') as fp:
    dic_achr = pickle.load(fp)
achr_id = dic_achr['achr_id']
achr_num = len(achr_id)
ori_iou_list = []
ja_iou_list = []
sa_iou_list = []
err_j_hmr_list = []
err_j_ja_list = []
err_j_sa_list = []


tr = trange(sta_num, desc='Bar desc', leave=True)
for test_num in tr:
 
    test_tuple = dataset[test_num]
    src_j = test_tuple[0]
    #src_a = test_tuple[1]
    src_img = test_tuple[2]
    verts = test_tuple[5]
    vert_norms = test_tuple[6]
    proc_para = test_tuple[7]
    all_sil = np.array(test_tuple[8])
    proj_sil = all_sil[:, :, 1]
    gt_sil = all_sil[:, :, 0]
    
    # for test joint
    joint_posi = np.array(test_tuple[9])
    joint_move = np.resize(np.array(test_tuple[3]), (10, 2))
    j_posi_gt = joint_posi+joint_move
    
    # predict joint
    joint_tsr = pdt_j.predict_batch(src_j)
    joint_para = np.array(joint_tsr.data.cpu())
    joint_para = np.concatenate((joint_para, np.zeros((10,1))),axis = 1)

    # apply scale
    joint_para = joint_para * 0.007# 0.007
    
    flat_point_list = [item for sublist in point_list for item in sublist]

    num_mj = len(point_list)
    j_list = []
    for i in range(num_mj):
        j_p_list = []
        for j in range(len(point_list[i])):
            j_p_list.append(verts[point_list[i][j]])
        j_list.append(sum(j_p_list)/len(j_p_list))

    new_jv = []
    ori_jv = []
    for i in range(len(j_list)):
        # make new joint verts
        for j in point_list[i]:
            new_jv.append(verts[j] + joint_para[i])
            ori_jv.append(verts[j])
    new_jv = np.array(new_jv)
    ori_jv = np.array(ori_jv)

    # joint deform
    fd_ja = fast_deform_dja(weight = 10.0)
    ja_verts = fd_ja.deform(np.asarray(verts), new_jv)

    
    # make src_a
    proj_sil_j = my_renderer.silhouette(verts = ja_verts)
    src_sil_j = np.zeros((224, 224, 2))
    src_a = np.zeros((200, 4, 32, 32))
    
    # make anchor posi
    anchor_verts = np.zeros((200, 3))
    for i in range(achr_num):
        anchor_verts[i, :] = ja_verts[achr_id[i], :]
    achr_posi = get_anchor_posi(anchor_verts)
    
    for i in range(len(achr_posi)):
        crop_sil = center_crop(proj_sil_j, achr_posi[i], 32)
        crop_img = center_crop(src_img, achr_posi[i], 32)
        crop_img = crop_img.astype(np.int)
        crop_img = crop_img - crop_img[15, 15, :]
        crop_img = np.absolute(crop_img)
        crop_img = crop_img.astype(np.float)/255.0
        src_a[i,0,:,:] = crop_sil
        src_a[i,1:4,:,:] = np.rollaxis(crop_img, 2, 0)

    # predict anchor
    achr_tsr = pdt_a.predict_batch(src_a)
    achr_para = np.array(achr_tsr.data.cpu())
    achr_para = achr_para * 0.003
    
    ori_av = []
    new_av = []
    for j in range(achr_num):
        ori_av.append(ja_verts[achr_id[j]])
        new_av.append(ja_verts[achr_id[j]] + 
                      vert_norms[achr_id[j]] * achr_para[j])
    ori_av = np.array(ori_av)
    new_av = np.array(new_av)
    
    # build active list of anchor
    contours, _ = cv2.findContours(proj_sil_j, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cim = np.zeros_like(proj_sil_j)
    cv2.drawContours(cim, contours, -1, 255, 1)
    cim = cv2.dilate(cim, kernel=np.ones((6, 6)))
    active_index = np.ones(len(achr_posi))
    for j in range(len(achr_posi)):
        ay = np.int(np.round(achr_posi[j][1]))
        ax = np.int(np.round(achr_posi[j][0]))
        if cim[ay, ax] == 0:
            active_index[j] = 0
    
    # anchor deform
    fd_sa = fast_deform_dsa(weight=1.0)
    sa_verts = fd_sa.deform(np.asarray(ja_verts), 
                            new_av,
                            active_index = active_index,
                           )
    
    # get sil after anchor deform
    sa_sil = my_renderer.silhouette(verts = sa_verts)
    
    # compute IoU and compare
    ori_iou = sil_iou(proj_sil, gt_sil)
    ja_iou = sil_iou(proj_sil_j, gt_sil)
    sa_iou = sil_iou(sa_sil, gt_sil)
    
    ori_iou_list.append(ori_iou)
    ja_iou_list.append(ja_iou)
    sa_iou_list.append(sa_iou)
    
    # compute error of joint
    j_posi_sa = get_joint_posi(sa_verts)
    err_j_sa = np.mean([np.linalg.norm(j_posi_sa[i]-j_posi_gt[i]) for i in range(10)])
    err_j_sa_list.append(err_j_sa)

    j_posi_ja = get_joint_posi(ja_verts)
    err_j_ja = np.mean([np.linalg.norm(j_posi_ja[i]-j_posi_gt[i]) for i in range(10)])
    err_j_ja_list.append(err_j_ja)
    
    j_posi_hmr = get_joint_posi(verts)
    err_j_hmr = np.mean([np.linalg.norm(j_posi_hmr[i]-j_posi_gt[i]) for i in range(10)])
    err_j_hmr_list.append(err_j_hmr)
    
# save eval values
output_dir = "./eval_report/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_dir + "wild_report.txt", "w") as f_txt:
    f_txt.write("# test_index, hmr_iou, hmd_j_iou, hmd_a(v)_iou, \r\n" + 
                "# hmr_2d_joint_err, hmd_j_2d_joint_err, " + 
                "hmd_a(v)_2d_joint_err\r\n")
    for i in range(sta_num):
        f_txt.write("%04d %f %f %f %f %f %f\r\n" \
                    % (i, ori_iou_list[i], ja_iou_list[i], sa_iou_list[i],  
                       err_j_hmr_list[i], err_j_ja_list[i], err_j_sa_list[i]))
    f_txt.write("mean %f %f %f %f %f %f\r\n" \
                % (np.mean(ori_iou_list[:sta_num]),
                   np.mean(ja_iou_list[:sta_num]),
                   np.mean(sa_iou_list[:sta_num]),
                   np.mean(err_j_hmr_list[:sta_num]),
                   np.mean(err_j_ja_list[:sta_num]),
                   np.mean(err_j_sa_list[:sta_num])))

# print results
print("IoU - hmr: %f, j: %f, a: %f" % (np.mean(ori_iou_list[:sta_num]),
                                       np.mean(ja_iou_list[:sta_num]),
                                       np.mean(sa_iou_list[:sta_num])))
print("Joint - hmr: %f, j: %f, a: %f" % (np.mean(err_j_hmr_list[:sta_num]),
                                         np.mean(err_j_ja_list[:sta_num]),
                                         np.mean(err_j_sa_list[:sta_num])))