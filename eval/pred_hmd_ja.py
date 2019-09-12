import numpy as np
import cv2, argparse, pickle, sys, PIL.Image
import openmesh as om
from tqdm import trange
sys.path.append("../src/")
from predictor import joint_predictor
from predictor import anchor_predictor
from mesh_edit import fast_deform_dja
from mesh_edit import fast_deform_dsa
from renderer import SMPLRenderer
from utility import center_crop
from utility import make_trimesh
from utility import get_anchor_posi
from utility import get_joint_posi

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num', type = int, required = True, 
                    help = 'data_num')
parser.add_argument('--set', type = str, required = True, 
                    help = 'recon or syn')
opt = parser.parse_args()

assert opt.set in ["recon", "syn"], \
       "set must be one of [recon, syn]"

data_num = int(opt.num)

pdt_j = joint_predictor("../demo/pretrained_model/pretrained_joint.pth")
pdt_a = anchor_predictor("../demo/pretrained_model/pretrained_anchor.pth")

renderer = SMPLRenderer(face_path="../predef/smpl_faces.npy")

faces = np.load("../predef/smpl_faces.npy")

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


tr = trange(data_num, desc='Bar desc', leave=True)
for test_ind in tr:

    src_img = np.array(PIL.Image.open("./eval_data/%s_set/input/%03d_img.png" \
                                      % (opt.set, test_ind)))

    #verts, cam, proc_para, std_img = hmr_pred.predict(src_img)
    with open ('./eval_data/%s_set/pred_save/pre_%03d.pkl' % \
               (opt.set, test_ind), 'rb') as fp:
        hmr_data = pickle.load(fp)
    verts = hmr_data['verts']
    cam = hmr_data['cam']
    proc_para = hmr_data['proc_para']
    std_img = hmr_data['std_img']

    mesh = make_trimesh(verts, faces, compute_vn = True)
    vert_norms = mesh.vertex_normals()

    # get proj sil
    proj_sil = renderer.silhouette(verts = verts,
                                   cam = cam,
                                   img_size = src_img.shape,
                                   norm = True)

    # make joint posi
    joint_posi = get_joint_posi(verts, point_list)

    sil_j = np.expand_dims(proj_sil, 2)
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
    proj_sil_j = renderer.silhouette(verts = ja_verts)
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
    
    # compute the achr movement
    ori_av = []
    new_av = []
    for j in range(achr_num):
        ori_av.append(ja_verts[achr_id[j]])
        new_av.append(ja_verts[achr_id[j]] + 
                      vert_norms[achr_id[j]] * achr_para[j])
    ori_av = np.array(ori_av)
    new_av = np.array(new_av)
    
    # build active list of anchor, added in 2018-10-30
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
    
    # visualize
    if False:
        ori_proj_img = renderer(verts = verts, img = src_img)
        joint_move_img = draw_vert_move(ori_jv, new_jv, src_img)
        achr_move_img = draw_vert_move(ori_av, new_av, src_img)
        ja_proj_img = renderer(verts = ja_verts, img = src_img)
        sa_proj_img = renderer(verts = sa_verts, img = src_img)
        final_prv_img = np.concatenate((ori_proj_img, 
                                        joint_move_img, 
                                        ja_proj_img, 
                                        achr_move_img, 
                                        sa_proj_img), axis = 1)
        show_img_arr(final_prv_img.astype(np.uint8))
    
    mesh_j = make_trimesh(ja_verts, faces)
    om.write_mesh("./eval_data/%s_set/pred_save/j_%03d.obj" % \
                  (opt.set, test_ind), mesh_j)
    mesh_a = make_trimesh(sa_verts, faces)
    om.write_mesh("./eval_data/%s_set/pred_save/a_%03d.obj" % \
                  (opt.set, test_ind), mesh_a)