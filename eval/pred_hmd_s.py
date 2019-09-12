from __future__ import print_function
import torch, PIL.Image, cv2, pickle, sys, argparse
import numpy as np
import openmesh as om
from tqdm import trange
sys.path.append("../src/")
from network import shading_net
import renderer as rd
from utility import subdiv_mesh_x4
from utility import CamPara
from utility import make_trimesh
from utility import flatten_naval
from utility import smpl_detoe


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num', type = int, required = True, 
                    help = 'data_num')
parser.add_argument('--set', type = str, required = True, 
                    help = 'recon or syn')
opt = parser.parse_args()

assert opt.set in ["recon", "syn"], \
       "set must be one of [recon, syn]"

# prepare
data_num = int(opt.num)
model_file = "../demo/pretrained_model/pretrained_shading.pth"
device = torch.device("cuda:0")
net_shading = shading_net().to(device).eval()
net_shading.load_state_dict(torch.load(model_file, map_location='cuda:0'))
renderer = rd.SMPLRenderer(face_path = 
                           "../predef/smpl_faces.npy")
cam_para = CamPara(K = np.array([[1000, 0, 224],
                                 [0, 1000, 224],
                                 [0, 0, 1]]))
with open ('../predef/exempt_vert_list.pkl', 'rb') as fp:
    exempt_vert_list = pickle.load(fp)

tr = trange(data_num, desc='Bar desc', leave=True)
for test_num in tr:
# read mesh
    mesh = om.read_trimesh("./eval_data/%s_set/pred_save/a_%03d.obj" % \
                           (opt.set, test_num))
    
    proj_sil = renderer.silhouette(verts = mesh.points())
    
    proj_sil_l = cv2.resize(proj_sil, dsize=(448, 448))
    proj_sil_l[proj_sil_l<0.5] = 0
    proj_sil_l[proj_sil_l>=0.5] = 1
    
    # load data
    src_img = np.array(PIL.Image.open("./eval_data/%s_set/input/%03d_img.png"%\
                                      (opt.set, test_num)))
    src_img_l = cv2.resize(src_img, dsize=(448, 448))
    input_arr = np.rollaxis(src_img_l, 2, 0)
    input_arr = np.expand_dims(input_arr, 0)
    input_arr = torch.tensor(input_arr).float().to(device)
    input_arr = input_arr/255.0

    proj_sil_l = np.expand_dims(proj_sil_l, 0)
    proj_sil_l = np.expand_dims(proj_sil_l, 0)
    proj_sil_l = torch.tensor(proj_sil_l)
    proj_sil_l = proj_sil_l.float().to(device)

    # predict
    pred = net_shading(input_arr, proj_sil_l)
    pred_depth = np.array(pred.data.cpu()[0][0])

    #show_img_arr(src_img)
    mesh = flatten_naval(mesh)

    # remove toes
    mesh = smpl_detoe(mesh)

    # subdivide the mesh to x4
    subdiv_mesh = subdiv_mesh_x4(mesh)

    # genrate boundary buffering mask
    sil_img = rd.render_sil(subdiv_mesh)
    bound_img = rd.render_bound(subdiv_mesh)

    radius = 10
    circ_template = np.zeros((radius*2+1, radius*2+1))
    for i in range(radius):
        cv2.circle(img = circ_template, 
                   center = (radius, radius), 
                   radius = i+2, 
                   color = (radius-i)*0.1, 
                   thickness = 2)
    
    img_size = bound_img.shape
    draw_img = np.zeros(img_size, dtype=np.float)
    draw_img = np.pad(draw_img, radius, 'edge')
    for y in range(img_size[0]):
        for x in range(img_size[1]):
            if bound_img[y, x] == 0:
                continue
            win = draw_img[y:y+2*radius+1, x:x+2*radius+1]
            win[circ_template>win] = circ_template[circ_template>win]
            draw_img[y:y+2*radius+1, x:x+2*radius+1] = win
    
    final_mask = sil_img - draw_img[10:10+img_size[0], 10:10+img_size[1]]
    final_mask[sil_img==0] = 0
    
    # apply bias
    d_max = np.max(pred_depth[pred_depth!=0])
    d_min = np.min(pred_depth[pred_depth!=0])
    bias = -(d_max - d_min)/2.
    pred_depth = pred_depth + bias
    
    # apply bright scale
    weight_map = np.dot(src_img_l[...,:3], [0.299, 0.587, 0.114])
    pred_depth = pred_depth * weight_map / 255.
    
    pred_depth = pred_depth * 0.001
    pred_depth = pred_depth * final_mask
    
    
    # project mesh to depth and merge with depth difference
    proj_depth, visi_map = rd.render_depth(subdiv_mesh, require_visi = True)

    # get all visible vertex index
    verts = subdiv_mesh.points()
    faces = subdiv_mesh.face_vertex_indices()
    visi_vert_inds = []
    for y in range(visi_map.shape[0]):
        for x in range(visi_map.shape[1]):
            f_ind = visi_map[y, x]
            if f_ind >= len(faces):
                continue
            else:
                fv = faces[f_ind]
                visi_vert_inds.append(fv[0])
                visi_vert_inds.append(fv[1])
                visi_vert_inds.append(fv[2])
    visi_vert_inds = set(visi_vert_inds)
    # filter out exempt version
    visi_vert_inds = list(set(visi_vert_inds).difference(exempt_vert_list))


    visi_vert_inds_m = []
    for i in visi_vert_inds:
        xy = cam_para.project(verts[i])
        x = int(round(xy[1]))
        y = int(round(xy[0]))
        if x<0 or y<0 or x>=448 or y>=448:
            continue
        if np.absolute(proj_depth[x, y] - verts[i,2])<0.01:
            visi_vert_inds_m.append(i)

    for i in visi_vert_inds_m:
        xy = cam_para.project(verts[i])
        x = int(round(xy[1]))
        y = int(round(xy[0]))
        depth = proj_depth[x, y] + pred_depth[x, y]
        #print(depth, verts[i])
        if depth>8.:
            continue
        verts[i][2] = depth

    deformed_mesh = make_trimesh(verts, faces)
    om.write_mesh("./eval_data/%s_set/pred_save/s_%03d.obj" % \
                  (opt.set, test_num), deformed_mesh)
