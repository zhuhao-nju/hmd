import numpy as np
import PIL.Image, open3d, sys, argparse
import openmesh as om
from tqdm import trange
from eval_functions import knnsearch
sys.path.append("../src/")
from renderer import SMPLRenderer
from renderer import render_depth
from utility import sil_iou
from eval_functions import get_hf_list

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num', type = int, default = 150, 
                    help = 'data_num')
parser.add_argument('--tgt', type = str, required = True, 
                    help = 'recon or syn')
opt = parser.parse_args()

set_name = "recon"
assert opt.tgt in ["j", "a", "s", "hmr"], \
       "tgt must be in in [j, a, s, hmr]"

data_num = int(opt.num)
error_list = []
iou_list = []
error_list_visi = []

my_renderer = SMPLRenderer(face_path="../predef/smpl_faces.npy")
hf_list = get_hf_list()


tr = trange(data_num, desc='Bar desc', leave=True)
for i in tr:
    # read results and gt mesh
    mesh_gt = om.read_trimesh("./eval_data/%s_set/gt/%03d.obj" % (set_name, i))
    mesh_tgt = om.read_trimesh("./eval_data/%s_set/pred_save/%s_%03d.obj" \
                               % (set_name, opt.tgt, i))
    
    verts_gt = mesh_gt.points()
    verts_tgt = mesh_tgt.points()
    
    # pick visible verts
    _, visi_gt = render_depth(mesh_gt, require_visi = True)
    _, visi_tgt = render_depth(mesh_tgt, require_visi = True)

    faces_gt = mesh_gt.face_vertex_indices()
    faces_tgt = mesh_tgt.face_vertex_indices()

    visi_gt = [visi_gt[x,y] for x in range(448) for y in range(448)]
    visi_gt = set(visi_gt)
    visi_gt.remove(4294967295)
    visi_gt = [faces_gt[fv][ind] for ind in range(3) for fv in visi_gt]
    visi_gt = list(set(visi_gt))
    
    visi_tgt = [visi_tgt[x,y] for x in range(448) for y in range(448)]
    visi_tgt = set(visi_tgt)
    visi_tgt.remove(4294967295)
    visi_tgt = [faces_tgt[fv][ind] for ind in range(3) for fv in visi_tgt]
    visi_tgt = list(set(visi_tgt))
    visi_tgt_fit = list(set(visi_tgt) - set(hf_list))
    
    verts_gt_visi = np.array([verts_gt[ind] for ind in visi_gt])
    verts_tgt_visi = np.array([verts_tgt[ind] for ind in visi_tgt])
    verts_tgt_visi_nhf = np.array([verts_tgt[ind] for ind in visi_tgt_fit])
    
    # compute scale
    ave_dist_gt = np.linalg.norm(np.mean(verts_gt_visi, axis=0))
    ave_dist_tgt = np.linalg.norm(np.mean(verts_tgt_visi, axis=0))
    scale = ave_dist_gt/ave_dist_tgt
    verts_tgt *= scale
    verts_tgt_visi *= scale
    verts_tgt_visi_nhf *= scale
    
    # do ICP
    source = open3d.PointCloud()
    target = open3d.PointCloud()
    source.points = open3d.Vector3dVector(verts_tgt_visi_nhf)
    target.points = open3d.Vector3dVector(verts_gt_visi)
    
    reg_p2p = open3d.registration_icp(source, 
                                      target, 
                                      10, 
                                      np.identity(4),
         open3d.TransformationEstimationPointToPoint(),
         open3d.ICPConvergenceCriteria(max_iteration = 10000),
                                     )
    trans_mat = reg_p2p.transformation
    verts_tgt_fit = np.zeros(verts_tgt.shape)
    verts_tgt_visi_fit = np.zeros(verts_tgt_visi.shape)
    
    for j in range(len(verts_tgt)):
        verts_tgt_fit[j] = np.dot(trans_mat[:3,:3], verts_tgt[j]) + trans_mat[:3,3]
    
    for j in range(len(verts_tgt_visi)):
        verts_tgt_visi_fit[j] = np.dot(trans_mat[:3,:3], verts_tgt_visi[j]) + trans_mat[:3,3]
    
    
    # compute error
    avergedist_tgt, _, crsp_ind_tgt = knnsearch(verts_tgt_fit, verts_gt)
    avergedist_tgt_visi, _, _ = knnsearch(verts_tgt_visi_fit, verts_gt_visi)
    
    error_list.append(avergedist_tgt)
    error_list_visi.append(avergedist_tgt_visi)
    
    # read gt sil
    sil_gt = np.array(PIL.Image.open("./eval_data/%s_set/input/%03d_sil.png" \
                                     % (set_name, i)))
    sil_gt[sil_gt<128] = 0
    sil_gt[sil_gt>=128] = 255
    
    # render result_sil
    sil_result = my_renderer.silhouette(verts = verts_tgt)
    iou = sil_iou(sil_result, sil_gt)
    iou_list.append(iou)
    
    crsp_verts_tgt = np.array([verts_tgt_fit[crsp_ind_tgt[j][0]]for j in range(len(crsp_ind_tgt))])
    
    diff_array_tgt = crsp_verts_tgt - verts_gt

    
print("mean %f %f %f\r\n" % \
            (np.mean(error_list),
             np.mean(error_list_visi),
             np.mean(iou_list)))

with open("./eval_report/%s_%s_report.txt" % (set_name, opt.tgt), "w") as f_txt:
    f_txt.write("# test_index, 3d_joint_err, 3d_joint_err_visi, iou\r\n")
    for i in range(data_num):
        f_txt.write("%03d %f %f %f\r\n" % \
                    (i, error_list[i], error_list_visi[i], iou_list[i]))    
    f_txt.write("mean %f %f %f\r\n" % \
                (np.mean(error_list),
                 np.mean(error_list_visi),
                 np.mean(iou_list)))
