import numpy as np
import PIL.Image
import cv2
import math
import skimage.draw
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import torch
# lighting functions
import lighting

# show image in Jupyter Notebook (work inside loop)
from io import BytesIO 
from IPython.display import display, Image
def show_img_arr(arr):
    im = PIL.Image.fromarray(arr)
    bio = BytesIO()
    im.save(bio, format='png')
    display(Image(bio.getvalue(), format='png'))

# write log in training phase
def take_notes(content, target_file, create_file = False):
    if create_file == True:
        f = open(target_file,"w")
    else:
        f = open(target_file,"a")
    f.write(content)
    f.close()
    return len(content)

# convenient for saving tensor to file as snapshot
def save_to_img(src, output_path_name, src_type = "tensor", channel_order="cwd", scale = 255):
    if src_type == "tensor":
        src_arr = np.asarray(src) * scale
    elif src_type == "array":
        src_arr = src*scale
    else:
        print("save tensor error, cannot parse src type.")
        return False
    if channel_order == "cwd":
        src_arr = (np.moveaxis(src_arr,0,2)).astype(np.uint8)
    elif channel_order == "wdc":
        src_arr = src_arr.astype(np.uint8)
    else:
        print("save tensor error, cannot parse channel order.")
        return False
    src_img = PIL.Image.fromarray(src_arr)
    src_img.save(output_path_name)
    return True

def save_batch_tensors(src_tensor, tgt_tensor, pred_tensor, output_name):
    src_arr = np.asarray(src_tensor)
    tgt_arr = np.asarray(tgt_tensor)
    pred_arr = np.asarray(pred_tensor)
    batch_size = src_arr.shape[0]
    chn = src_arr.shape[1]
    height = src_arr.shape[2]
    width = src_arr.shape[3]
    board_arr = np.zeros((chn, height*batch_size, width*3))
    for j in range(batch_size):
        board_arr[:,j*height:(j+1)*height,0:width] = src_arr[j]
        board_arr[:,j*height:(j+1)*height,width:2*width] = tgt_arr[j]
        board_arr[:,j*height:(j+1)*height,2*width:3*width] = pred_arr[j]
    save_to_img(board_arr, output_name, src_type = "array")
    
# camera project and inv-project
class CamPara():
    def __init__(self, K=None, Rt=None):
        img_size = [200,200]
        if K is None:
            K = np.array([[500, 0, 112],
                          [0, 500, 112],
                          [0, 0, 1]])
        else:
            K = np.array(K)
        if Rt is None:
            Rt = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])
        else:
            Rt = np.array(Rt)
        R = Rt[:,:3]
        t = Rt[:,3]
        self.cam_center = -np.dot(R.transpose(),t)
        
        # compute projection and inv-projection matrix
        self.proj_mat = np.dot(K, Rt)
        self.inv_proj_mat = np.linalg.pinv(self.proj_mat)

        # compute ray directions of camera center pixel
        c_uv = np.array([float(img_size[0])/2+0.5, float(img_size[1])/2+0.5])
        self.center_dir = self.inv_project(c_uv)
            
    def get_camcenter(self):
        return self.cam_center
    
    def get_center_dir(self):
        return self.center_dir
    
    def project(self, p_xyz):
        p_xyz = np.double(p_xyz)
        p_uv_1 = np.dot(self.proj_mat, np.append(p_xyz, 1))
        if p_uv_1[2] == 0:
            return 0
        p_uv = (p_uv_1/p_uv_1[2])[:2]
        return p_uv
    
    # inverse projection, if depth is None, return a normalized direction
    def inv_project(self, p_uv, depth=None):
        p_uv = np.double(p_uv)
        p_xyz_1 = np.dot(self.inv_proj_mat, np.append(p_uv, 1))
        if p_xyz_1[3] == 0:
            return 0
        p_xyz = (p_xyz_1/p_xyz_1[3])[:3]
        p_dir = p_xyz - self.cam_center
        p_dir = p_dir / np.linalg.norm(p_dir)
        if depth is None:
            return p_dir
        else:
            real_xyz = self.cam_center + p_dir * depth
            return real_xyz
        
# for photometric loss       
def photometricLossgray(colorImg_gray, depthImg, albedoImg_gray, 
                        mask, lighting_est, device, K, thres):
    
    N,C,H,W = colorImg_gray.size()
    
    # color loss
    normals, _ = lighting.depthToNormalBatch(depthImg, device, K, thres)
    SHs     = lighting.normalToSHBatch(normals,device)
    
    SHs    = torch.reshape(SHs, (N, H*W, 9))
    lighting_est = torch.reshape(lighting_est, (N, 9, 1))
    
    #SHs to [B, H*W,9] lighting [B, 9, 1] --[N, H*W] --[B,H,W,1]             
    color_shading = torch.bmm(SHs, lighting_est) # N H*W 1   
    color_shading = torch.reshape(color_shading, (N, H, W))
    
    mask1 = torch.reshape(mask[:,0,:,:], (N,H,W)) # one layer mask
    color_pre  = mask1 * (color_shading * albedoImg_gray) # N*H*W
    colorImg_gray_mask = mask1 * colorImg_gray # mask
    
    colorloss = F.l1_loss(color_pre, colorImg_gray_mask) # NHW size directly
        
    return colorloss, color_pre

# come from hmr-src/util/image.py
def scale_and_crop(image, scale, center, img_size):
    image_scaled, scale_factors = resize_img(image, scale)
    # Swap so it's [x, y]
    scale_factors = [scale_factors[1], scale_factors[0]]
    center_scaled = np.round(center * scale_factors).astype(np.int)

    margin = int(img_size / 2)
    image_pad = np.pad(
        image_scaled, ((margin, ), (margin, ), (0, )), mode='edge')
    center_pad = center_scaled + margin
    # figure out starting point
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    # crop:
    crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    proc_param = {
        'scale': scale,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': img_size
    }
    return crop, proc_param

def get_line_length(pixel_list):
    if len(pixel_list) <= 1:
        return 0
    max_x, min_x = pixel_list[0][0], pixel_list[0][0]
    max_y, min_y = pixel_list[0][1], pixel_list[0][1]
    for i in range(len(pixel_list)):
        if pixel_list[i][0]>max_x:
            max_x = pixel_list[i][0]
        elif pixel_list[i][0]<min_x:
            min_x  = pixel_list[i][0]
        if pixel_list[i][1]>max_y:
            max_y = pixel_list[i][1]
        elif pixel_list[i][1]<min_y:
            min_y  = pixel_list[i][1]
    l_x = max_x - min_x
    l_y = max_y - min_y
    length = len(pixel_list) * np.linalg.norm([l_x, l_y]) / max(l_x, l_y)
    return length

# compute the distance between anchor points and silhouette boundary, 
# (the model occluded parts are filtered out)
def measure_achr_dist(achr_verts, 
                      achr_normals, 
                      gt_sil, 
                      proj_sil,
                      hd_size = (1000, 1000),
                      max_dist = 0.05,
                      anchor_num = 200,
                     ):
    # parse projected image to get distance for each anchor
    if gt_sil.shape is not hd_size:
        gt_sil = cv2.resize(gt_sil, dsize=(1000, 1000), interpolation=cv2.INTER_LINEAR)
    if proj_sil.shape is not hd_size:
        proj_sil = cv2.resize(proj_sil, dsize=(1000, 1000), interpolation=cv2.INTER_LINEAR)
    
    proj_img_out = np.zeros((hd_size))
    proj_img_in = np.zeros((hd_size))
    cam_para = CamPara(K = np.array([[2232.142857, 0, 500],
                                     [0, 2232.142857, 500],
                                     [0,       0,       1]]))
    # project vectors to image as index-valued lines
    # make start_point list and end_out_point list, end_in_point list
    start_point_list = []
    end_out_point_list = []
    end_in_point_list = []
    for i in range(len(achr_verts)):
        xy = cam_para.project(achr_verts[i])
        x = int(xy[0]+0.5)
        y = int(xy[1]+0.5)
        if x<0 or y<0 or x>=hd_size[1] or y>=hd_size[0]:
            continue
        
        uv = cam_para.project(achr_verts[i] + achr_normals[i]*max_dist)
        u = int(uv[0]+0.5)
        v = int(uv[1]+0.5)
        if u<0 or v<0 or u>=hd_size[1] or v>=hd_size[0]:
            continue
        
        ab = cam_para.project(achr_verts[i] - achr_normals[i]*max_dist)
        a = int(ab[0]+0.5)
        b = int(ab[1]+0.5)
        if a<0 or b<0 or a>=hd_size[1] or b>=hd_size[0]:
            continue
        
        r_out, c_out = skimage.draw.line(y,x,v,u)
        r_in, c_in = skimage.draw.line(y,x,b,a)
        
        # draw img out and in
        proj_img_out[r_out, c_out] = i+1
        proj_img_in[r_in, c_in] = i+1
        
    proj_img_out[proj_sil>=128] = 0
    proj_img_out[gt_sil<128] = 0
    proj_img_in[proj_sil<128] = 0
    proj_img_in[gt_sil>=128] = 0
    
    # build pixel map for efficiently using get_line_length()
    pixel_map_in = [[] for i in range(anchor_num)]
    pixel_map_out = [[] for i in range(anchor_num)]
    for x in range(hd_size[1]):
        for y in range(hd_size[0]):
            if proj_img_in[x, y] > 0:
                pixel_map_in[int(proj_img_in[x, y])-1].append([x,y])
            if proj_img_out[x, y] > 0:
                pixel_map_out[int(proj_img_out[x, y])-1].append([x,y])

    # compute index list
    index_list = [0] * len(achr_verts)
    for i in range(anchor_num):
        length_in = get_line_length(pixel_map_in[i]) #len(pixel_map_in[i])
        length_out = get_line_length(pixel_map_out[i]) #len(pixel_map_out[i])
        if length_in>length_out:
            index_list[i] = -length_in
        elif length_out>length_in:
            index_list[i] = length_out
        else:
            index_list[i] = 0
        
            
        #if length_in<2 and length_out>2:
        #    index_list[i] = length_out
        #elif length_in>2 and length_out<2:
        #    index_list[i] = -length_in
        #elif length_in>2 and length_out >2:
        #    if start_exist_out_list[i] == True:
        #        index_list[i] = length_out
        #    elif start_exist_in_list[i] == True:
        #        index_list[i] = -length_in
        #    else:
        #        index_list[i] = 0
        #else:
        #    index_list[i] = 0
        
        
        #if length_out >= length_in:
        #    index_list[i] = length_out
        #else:
        #    index_list[i] = -length_in
    return index_list


# draw vert moving vector in an image
def draw_vert_move(ori_achrs, new_achrs, bg_img = None):
    if len(ori_achrs) != len(new_achrs):
        print("ERROR: length not matched in draw_vert_move()")
        return False
    if bg_img is None:
        bg_img = np.zeros((224,224,3))
    else:
        bg_img = bg_img.copy()
    cam_para = CamPara()
    img_size = bg_img.shape
    for i in range(len(ori_achrs)):
        xy = cam_para.project(ori_achrs[i])
        x = int(xy[0]+0.5)
        y = int(xy[1]+0.5)
        if x<0 or y<0 or x>=img_size[1] or y>=img_size[0]:
            continue
        uv = cam_para.project(new_achrs[i])
        u = int(uv[0]+0.5)
        v = int(uv[1]+0.5)
        if u<0 or v<0 or u>=img_size[1] or v>=img_size[0]:
            continue
        r, c = skimage.draw.line(y,x,v,u)
        if(len(r)<3):
            continue
        bg_img[r, c, :] = np.array([0, 0, 255])
        bg_img[y, x, :] = np.array([0, 255, 0])
    return bg_img


# display loss, support multiple draw
class loss_board():
    def __init__(self):
        self.color_list = ("b","g","r","c","m","y","k","w")
        self.color_id = 0
        self.draw_num = 0
        self.fig, self.ax = plt.subplots()
        self.data = []
        
    def draw(self, loss_file, kn_smth = 0):

        # read file
        f = open(loss_file, "r")
        
        # skip file header
        f.readline()
        
        # make data list
        ctt = f.read().split()
        num = len(ctt)/3
        data_list = []
        for i in range(num):
            data_list.append(float(ctt[i*3]))
        
        # smooth if neccessary
        if kn_smth != 0:
            data_list_smth = []
            for i in range(num):
                d_sum = 0
                count = 0
                for j in range(i - kn_smth, i + kn_smth + 1):
                    if j<0 or j>= num:
                        continue
                    else:
                        d_sum += data_list[j]
                        count += 1
                data_list_smth.append(d_sum/count)
        data_list = data_list_smth
        self.data.append(data_list)
        self.ax.plot(data_list, color = self.color_list[self.color_id])
        
        self.draw_num += 1
        self.color_id = (self.draw_num) % len(self.color_list)
        
        
    def show(self):
        txt_ctt = ""
        for i in range(self.draw_num):
            if i == 0:
                txt_ctt += "%d -- %s" % \
                           (i,self.color_list[i%len(self.color_list)])
            else:
                txt_ctt += "\n%d -- %s" % \
                           (i, self.color_list[i%len(self.color_list)])
        plt.text(0.9, 0.85, 
                 txt_ctt, 
                 transform = self.ax.transAxes, 
                 size=10, 
                 ha="center", 
                 va="center", 
                 bbox=dict(boxstyle="round",color="silver")
                )
        
        plt.show()
        
    def get_list(self):
        return self.data



def get_joint_move(verts, lsp_joint, proc_para, mesh_joint, unseen_mode=False):
    scale = proc_para["scale"]
    img_size = proc_para["img_size"]
    bias = np.array([img_size/2, img_size/2]) - proc_para["start_pt"]    
    point_list = mesh_joint["point_list"]
    index_map = mesh_joint["index_map"]
    
    flat_point_list = [item for sublist in point_list for item in sublist]
    
    num_mj = len(point_list)
    j_list = []
    for i in range(num_mj):
        j_p_list = []
        for j in range(len(point_list[i])):
            j_p_list.append(verts[point_list[i][j]])
        j_list.append(sum(j_p_list)/len(j_p_list))

    new_joint_verts = []
    ori_joint_verts = []
    cam_para = CamPara()
    joint_move = []
    joint_posi = []
    
    for i in range(len(j_list)):
        src_yx = cam_para.project(j_list[i])
        src_y = src_yx[0]
        src_x = src_yx[1]
        joint_posi.append(src_yx.tolist())
        if len(index_map[i]) == 1:
            tgt_x = lsp_joint[1,index_map[i][0]]
            tgt_y = lsp_joint[0,index_map[i][0]]
            unseen_label = lsp_joint[2,index_map[i][0]]
        elif len(index_map[i]) == 2:
            tgt_x = (lsp_joint[1,index_map[i][0]] + 
                     lsp_joint[1,index_map[i][1]]) / 2
            tgt_y = (lsp_joint[0,index_map[i][0]] + 
                     lsp_joint[0,index_map[i][1]]) / 2
            unseen_label = lsp_joint[2,index_map[i][0]] * \
                          lsp_joint[2,index_map[i][1]]
        tgt_y = tgt_y*scale + bias[0]
        tgt_x = tgt_x*scale + bias[1]
        
        #perspect_scale = j_list[i][2]/5. # proved to be unnecessary
        joint_move_t = np.array([tgt_y - src_y, tgt_x - src_x, 0])
        
        # many joints in LSPET/COCO are valid, filter them out using this label
        if unseen_mode is True and unseen_label <= 0:
            joint_move_t = joint_move_t*0
            
        joint_move.append(joint_move_t[:2])
        # make new joint verts
        for j in point_list[i]:
            new_joint_verts.append(verts[j] + joint_move_t*0.007)
            ori_joint_verts.append(verts[j])

    joint_move = np.array(joint_move)
    joint_posi = np.array(joint_posi)
    joint_posi[joint_posi<0] = 0
    joint_posi[joint_posi>(img_size-1)] = img_size-1
    new_joint_verts = np.array(new_joint_verts)        
    ori_joint_verts = np.array(ori_joint_verts)
    
    return new_joint_verts, ori_joint_verts, joint_move, joint_posi


def get_joint_posi(verts, 
                   point_list = [],
                   j2or3 = 2,
                   img_size = 224,
                   K = None,
                   Rt = None):
    
    if point_list == []:
        # read joint indexes
        with open ('../predef/mesh_joint_list.pkl', 'rb') as fp:
            item_dic = pickle.load(fp)
        point_list = item_dic["point_list"]
    
    num_mj = len(point_list)
    joint3d_list = []
    for i in range(num_mj):
        j_p_list = []
        for j in range(len(point_list[i])):
            j_p_list.append(verts[point_list[i][j]])
        joint3d_list.append(sum(j_p_list)/len(j_p_list))
    
    if j2or3 == 3:
        return joint3d_list
    elif j2or3 == 2:
        cam_para = CamPara(K = K, Rt = Rt)
        joint2d_list = []
        for i in range(len(joint3d_list)):
            src_yx = cam_para.project(joint3d_list[i])
            joint2d_list.append(src_yx.tolist())
        joint2d_list = np.array(joint2d_list)
        joint2d_list[joint2d_list<0] = 0
        joint2d_list[joint2d_list>(img_size-1)] = img_size-1
        return joint2d_list
    else:
        print("WARN: wrong j2or3 variable in get_joint_posi()")
        return []
    
# get anchor movement
def get_achr_move(gt_sil, verts, vert_norms, proj_sil):
    with open ('../predef/dsa_achr.pkl', 'rb') as fp:
        dic_achr = pickle.load(fp)
    achr_id = dic_achr['achr_id']

    achr_num = len(achr_id)
    ori_achr_verts = []
    achr_norms = []
    for j in range(achr_num):
        ori_achr_verts.append(verts[achr_id[j]])
        achr_norms.append(vert_norms[achr_id[j]])
    ori_achr_verts = np.array(ori_achr_verts)
    achr_norms = np.array(achr_norms)
    
    # predict anchor_move of anchor point
    achr_move = measure_achr_dist(ori_achr_verts, 
                                  achr_norms, 
                                  gt_sil, 
                                  proj_sil)
    achr_move = np.array(achr_move)
    diff = achr_move * 0.003
    # make new_achr_verts 
    new_achr_verts = []
    for j in range(achr_num):
        new_achr_verts.append(ori_achr_verts[j] + achr_norms[j] * diff[j])
    new_achr_verts = np.array(new_achr_verts)

    return new_achr_verts, ori_achr_verts, achr_move

# compute Intersection over Union
def sil_iou(src_sil, tgt_sil):
    # transfer to int array
    src_sil = np.array(src_sil).astype(np.int)
    tgt_sil = np.array(tgt_sil).astype(np.int)
    
    # check channel numbers
    if len(src_sil.shape)>2 or len(tgt_sil.shape)>2:
        print("ERROR: input channel of sil_iou is more than two.")
        return False
    
    # threshold
    src_sil[src_sil!=0] = 1
    tgt_sil[tgt_sil!=0] = 1
    
    # compute IoU
    sil_I = src_sil - tgt_sil
    sil_I[sil_I!=0] = 1
    sil_U = src_sil + tgt_sil
    sil_U[sil_U!=0] = 1
    
    iou = 1. - float(np.sum(sil_I))/float(np.sum(sil_U))
    return iou

# for smpl model random joint deform
import random
from mesh_edit import fast_deform_dja
class random_joint_deform():
    def __init__(self,
                 predef_vert = True,
                 verts = [],
                 max_dist = 0.1):
        
        self.predef_vert = predef_vert
        self.max_dist = max_dist
        self.fd_j = fast_deform_dja(weight = 10.0)
        
        # read joint index list
        with open ('../predef/mesh_joint_list.pkl', 'rb') as fp:
            item_dic = pickle.load(fp)
        self.point_list = item_dic["point_list"]
        
        if self.predef_vert == True:
            if verts == []:
                print("ERROR: no predefine verts found when initialize RJD")
            else:
                self.verts = verts
        
    def __call__(self,
                 verts = []):
        if self.predef_vert == False:
            if verts == []:
                print("ERROR: no verts found when run RJD")
                return False
            self.verts = verts
        
        new_joint_verts = []
        ori_joint_verts = []
        for i in range(len(self.point_list)):
            joint_move = np.array([random.random() - 0.5, 
                                   random.random() - 0.5, 
                                   0])
            if i == 5:
                # joint weight decrease for hip 
                j_scale = self.max_dist * 0.1
            else:
                j_scale = self.max_dist
                
            for j in self.point_list[i]:
                new_joint_verts.append(self.verts[j] + joint_move*j_scale)
                ori_joint_verts.append(self.verts[j])
        ori_joint_verts = np.array(ori_joint_verts)
        new_joint_verts = np.array(new_joint_verts)
        
        new_verts = self.fd_j.deform(np.asarray(self.verts), 
                                     new_joint_verts)
        
        return new_verts, ori_joint_verts, new_joint_verts
    

# get silhouette boundingbox
def get_sil_bbox(sil, margin = 0):
    if len(sil.shape)>2:
        sil = sil[:,:,0]
    sil_col = np.sum(sil,1)
    sil_row = np.sum(sil,0)
    y_min = np.argmax(sil_col>0)
    y_max = len(sil_col) - np.argmax(np.flip(sil_col, 0)>0)
    x_min = np.argmax(sil_row>0)
    x_max = len(sil_row) - np.argmax(np.flip(sil_row, 0)>0)
    if margin != 0:
        y_min -= margin
        x_min -= margin
        y_max += margin
        x_max += margin
    return y_min, y_max, x_min, x_max

# come from hmr-src/util/image.py
def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor

import openmesh
# Compose verts and faces to openmesh TriMesh
def make_trimesh(verts, faces, compute_vn = True):
    # if vertex index starts with 1, make it start with 0
    if np.min(faces) == 1:
        faces = np.array(faces)
        faces = faces - 1
    
    # make a mesh
    mesh = openmesh.TriMesh()

    # transfer verts and faces
    for i in range(len(verts)):
        mesh.add_vertex(verts[i])
    for i in range(len(faces)):
        a = mesh.vertex_handle(faces[i][0])
        b = mesh.vertex_handle(faces[i][1])
        c = mesh.vertex_handle(faces[i][2])
        mesh.add_face(a,b,c)

    # compute vert_norms
    if compute_vn is True:
        mesh.request_vertex_normals()
        mesh.update_normals()

    return mesh

# get shifted verts
def shift_verts(proc_param, verts, cam):
    img_size = proc_param['img_size']
    cam_s = cam[0]
    cam_pos = cam[1:]
    flength = 500.
    tz = flength / (0.5 * img_size * cam_s)
    trans = np.hstack([cam_pos, tz])
    vert_shifted = verts + trans
    return vert_shifted

# transform mpii joints to standard lsp definition
# for ALL at one time
def transform_mpii_joints(joints):
    num = joints.shape[2]
    joints_t = np.zeros((3, 14, num))
    joints_t[:,0:6,:] = joints[:,0:6,:] # lower limbs
    joints_t[:,6:12,:] = joints[:,10:16,:] # upper limbs
    joints_t[:,12,:] = joints[:,8,:] # head
    joints_t[:,13,:] = joints[:,9,:] # neck
    
    
    # head compensation
    joints_t[:2,13,:] = joints_t[:2,13,:]*0.8 + joints_t[:2,12,:]*0.2
    
    # anckle compensation
    joints_t[:2,5,:] = joints_t[:2,5,:]*0.95 + joints_t[:2,4,:]*0.05
    joints_t[:2,0,:] = joints_t[:2,0,:]*0.95 + joints_t[:2,1,:]*0.05
    
    return joints_t

# transform coco joints to standard lsp definition
# for ONLY one tuple
def transform_coco_joints(joints):
    joints = np.transpose(joints)
    joints_t = np.zeros((3, 14))
    joints_t[:,0] = joints[:,16]  # Right ankle
    joints_t[:,1] = joints[:,14]  # Right knee
    joints_t[:,2] = joints[:,12]  # Right hip
    joints_t[:,3] = joints[:,11]  # Left hip
    joints_t[:,4] = joints[:,13]  # Left knee
    joints_t[:,5] = joints[:,15]  # Left ankle
    joints_t[:,6] = joints[:,10]  # Right wrist
    joints_t[:,7] = joints[:,8]  # Right elbow
    joints_t[:,8] = joints[:,6]  # Right shoulder
    joints_t[:,9] = joints[:,5]  # Left shoulder
    joints_t[:,10] = joints[:,7]  # Left elbow
    joints_t[:,11] = joints[:,9]  # Left wrist
    joints_t[:,12] = np.array([-1, -1, 0])  # Neck
    joints_t[:,13] = np.array([-1, -1, 0])  # Head top
    
    return joints_t

# transform h36m joints to standard lsp definition
# for ONLY one tuple
def transform_h36m_joints(joints):
    joints = np.resize(joints, (32, 2)).transpose()
    joints_t = np.ones((3, 14))
    joints_t[:2,0] = joints[:,3]  # Right ankle
    joints_t[:2,1] = joints[:,2]  # Right knee
    joints_t[:2,2] = joints[:,1]  # Right hip
    joints_t[:2,3] = joints[:,6]  # Left hip
    joints_t[:2,4] = joints[:,7]  # Left knee
    joints_t[:2,5] = joints[:,8]  # Left ankle
    joints_t[:2,6] = joints[:,27]  # Right wrist
    joints_t[:2,7] = joints[:,26]  # Right elbow
    joints_t[:2,8] = joints[:,25]  # Right shoulder
    joints_t[:2,9] = joints[:,17]  # Left shoulder
    joints_t[:2,10] = joints[:,18]  # Left elbow
    joints_t[:2,11] = joints[:,19]  # Left wrist
    joints_t[:2,12] = joints[:,13]  # Neck
    joints_t[:2,13] = joints[:,15]  # Head top
    
    # anckle compensation
    joints_t[:2,5] = joints_t[:2,5]*0.85 + joints_t[:2,4]*0.15
    joints_t[:2,0] = joints_t[:2,0]*0.85 + joints_t[:2,1]*0.15
    
    return joints_t

# draw sil from seg_points
def points2sil(seg_points, sil_shape):
    seg_points = np.array(seg_points).astype(np.int32)
    if len(seg_points.shape) == 1:
        p_num = len(seg_points)/2
        seg_points = np.resize(seg_points, (p_num, 2))
    sil = np.zeros(sil_shape)
    cv2.fillPoly(sil, [seg_points], (255), lineType=8)
    return sil
    
def pad_arr(arr, pad):
    if len(arr.shape) == 3:
        pad_img = np.pad(arr.tolist(), ((pad,pad),(pad,pad),(0,0)), "edge")
    elif len(arr.shape) == 2:
        pad_img = np.pad(arr.tolist(), ((pad,pad),(pad,pad)), "edge")
    else:
        print("ERROR: cannot understand arr structure in func: pad_arr")
        pad_img = False
    pad_img = pad_img.astype(arr.dtype)
    return pad_img

def center_crop(arr, center, size = 64):
    center = np.asarray(center)
    img_size = arr.shape[0]
    center[center<0] = 0
    center[center>(img_size-1)] = img_size-1
    half_size = int(size/2.0)
    arr_pad = pad_arr(arr, half_size)
    center = np.round(center)
    start_pt = (np.round(center)).astype(np.int)
    end_pt = (np.round(center) + half_size*2).astype(np.int)
    #print(start_pt[0],end_pt[0])
    if len(arr.shape) == 3:
        return arr_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    else:
        return arr_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]]
    
def center_crop_2dsize(arr, center, size):
    center = np.array(center)
    size = np.array(size)
    img_size = arr.shape[0]
    center[center<0] = 0
    center[center>(img_size-1)] = img_size-1
    half_size = (size/2.0).astype(np.int)
    max_hs = np.max(half_size)
    arr_pad = pad_arr(arr, max_hs)
    center = np.round(center)
    start_pt = (np.round(center) - half_size + max_hs).astype(np.int)
    end_pt = (np.round(center) + half_size + max_hs).astype(np.int)
    #print(start_pt[0],end_pt[0])
    if len(arr.shape) == 3:
        return arr_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    else:
        return arr_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]]
    
# for visualizing predict window in images
def draw_rect(img_arr, center, size=64, color=[255,0,0]):
    ori_dtype = img_arr.dtype
    img_arr = img_arr.astype(np.float)
    half_size = int(size/2.0)
    center = np.round(center)
    start_pt = (np.round(center) - half_size).astype(np.int)
    end_pt = (np.round(center) + half_size).astype(np.int)
    cv2.rectangle(img_arr, tuple(start_pt), tuple(end_pt), color)
    return img_arr.astype(ori_dtype)

# for visualizing predict window in images
def draw_joints_rect(img_arr, joint_posi, ratio = 1):
    ori_dtype = img_arr.dtype
    joint_num = len(joint_posi)
    seed_arr = np.array([range(1,255,255/joint_num)]).astype(np.uint8)
    color_list = cv2.applyColorMap(seed_arr, cv2.COLORMAP_RAINBOW)[0]
    draw_arr = img_arr.astype(np.float)
    for i in range(joint_num):
        draw_arr = draw_rect(draw_arr, joint_posi[i], 
                             color = color_list[i].tolist())
    if ratio < 1:
        draw_arr = draw_arr*ratio + img_arr.astype(np.float)*(1-ratio)
    return draw_arr.astype(ori_dtype)

# for visualizing predict window in images
def draw_anchors_rect(img_arr, anchor_posi, sample = 1, ratio = 1):
    ori_dtype = img_arr.dtype
    joint_num = len(anchor_posi)
    seed_arr = np.array([range(1,255,255/joint_num)]).astype(np.uint8)
    color_list = cv2.applyColorMap(seed_arr, cv2.COLORMAP_RAINBOW)[0]
    draw_arr = img_arr.astype(np.float)
    for i in range(joint_num):
        if (i%sample)!=0:
            continue
        draw_arr = draw_rect(draw_arr, anchor_posi[i], 
                             size = 32,
                             color = color_list[i].tolist())
    if ratio < 1:
        draw_arr = draw_arr*ratio + img_arr.astype(np.float)*(1-ratio)    
    return draw_arr.astype(ori_dtype)

# write OBJ from vertex
# not tested yet
def verts2obj(out_verts, filename):
    vert_num = len(out_verts)
    faces = np.load("../predef/smpl_faces.npy")
    face_num = len(faces)
    with open(filename, 'w') as fp:
        for j in range(vert_num):
            fp.write( 'v %f %f %f\n' % ( out_verts[j,0], out_verts[j,1], out_verts[j,2]) )
        for j in range(face_num):
            fp.write( 'f %d %d %d\n' %  (faces[j,0]+1, faces[j,1]+1, faces[j,2]+1) )
    PIL.Image.fromarray(src_img.astype(np.uint8)).save("./output/src_img_%d.png" % test_num)
    return True

# compute anchor_posi from achr_verts
def get_anchor_posi(achr_verts):
    cam_para = CamPara()
    achr_num = len(achr_verts)
    achr_posi = np.zeros((achr_num, 2))
    for i in range(achr_num):
        achr_posi[i] = cam_para.project(achr_verts[i])
    return achr_posi

# here supplement a post-processing for seg, to filter out
# the some objects containing less than min_pixel pixels
def refine_sil(sil, min_pixel):
    if len(sil.shape)==3:
        sil = sil[:,:,0]
        c3 = True
    else:
        c3 = False
    sil[sil>0] = 255
    
    nb_components, output, stats, centroids = \
            cv2.connectedComponentsWithStats(sil, connectivity = 8)
    
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    refined_sil = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_pixel:
            refined_sil[output == i + 1] = 255
            
    if c3 is True:
        refined_sil = np.stack((refined_sil,)*3, -1)
    return refined_sil
  
    

# subdivide mesh to 4 times faces    
import openmesh as om
def subdiv_mesh_x4(mesh):
    # get original vertex list
    verts = mesh.points()
    verts_list = verts.tolist()
    verts_num = len(verts_list)

    # make matrix to represent the id of each two verts
    new_vert_dic = np.zeros((verts_num, verts_num), dtype = np.int)

    # add vertexes
    crt_id = verts_num
    for e in mesh.edge_vertex_indices():
        new_vert_dic[e[0], e[1]] = crt_id
        new_vert_dic[e[1], e[0]] = crt_id
        verts_list.append((verts[e[0]] + verts[e[1]])/2.)
        crt_id += 1

    faces_list = []

    # add faces
    for f in mesh.face_vertex_indices():
        v1 = f[0]
        v2 = f[1]
        v3 = f[2]
        v4 = new_vert_dic[v1, v2]
        v5 = new_vert_dic[v2, v3]
        v6 = new_vert_dic[v3, v1]
        faces_list.append([v1, v4, v6])
        faces_list.append([v4, v2, v5])
        faces_list.append([v6, v5, v3])
        faces_list.append([v4, v5, v6])

    # make new mesh
    subdiv_mesh = make_trimesh(verts_list, faces_list,  compute_vn = False)
    return subdiv_mesh

# remove toes from smpl mesh model
def smpl_detoe(mesh):
    d_inds = [5506, 5507, 5508, 5509, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 
              5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525, 5526, 5527, 
              5528, 5529, 5530, 5531, 5532, 5533, 5534, 5535, 5536, 5537, 5538, 
              5539, 5540, 5541, 5542, 5543, 5544, 5545, 5546, 5547, 5548, 5549, 
              5550, 5551, 5552, 5553, 5554, 5555, 5556, 5557, 5558, 5559, 5560, 
              5561, 5562, 5563, 5564, 5565, 5566, 5567, 5568, 5569, 5570, 5571, 
              5572, 5573, 5574, 5575, 5576, 5577, 5578, 5579, 5580, 5581, 5582, 
              5583, 5584, 5585, 5586, 5587, 5588, 5589, 5590, 5591, 5592, 5593, 
              5594, 5595, 5596, 5597, 5598, 5599, 5600, 5601, 5602, 5603, 5604, 
              5605, 5606, 5607, 5608, 5609, 5610, 5611, 5612, 5613, 5614, 5615, 
              5616, 5617, 5618, 5619, 5620, 5621, 5622, 5623, 5624, 5625, 5626, 
              5627, 5628, 5629, 5630, 5631, 5632, 5633, 5634, 5635, 5636, 5637, 
              5638, 5639, 5640, 5641, 5642, 5643, 5644, 5645, 5646, 5647, 5648, 
              5649, 5650, 5651, 5652, 5653, 5654, 5655, 5656, 5657, 5658, 5659, 
              5660, 5661, 5662, 5663, 5664, 5665, 5666, 5667, 5668, 5669, 5670, 
              5671, 5672, 5673, 5674, 5675, 5676, 5677, 5678, 5679, 5680, 5681, 
              5682, 5683, 5684, 5685, 5686, 5687, 5688, 5689, 5690, 5691, 5692, 
              5693, 5694, 5695, 5696, 5697, 5698, 5699, 5700, 5701, 5702, 5703, 
              5704, 5705, 12394, 12395, 12396, 12397, 12398, 12399, 12400, 12401, 
              12402, 12403, 12404, 12405, 12406, 12407, 12408, 12409, 12410, 
              12411, 12412, 12413, 12414, 12415, 12416, 12417, 12418, 12419, 
              12420, 12421, 12422, 12423, 12424, 12425, 12426, 12427, 12428, 
              12429, 12430, 12431, 12432, 12433, 12434, 12435, 12436, 12437, 
              12438, 12439, 12440, 12441, 12442, 12443, 12444, 12445, 12446, 
              12447, 12448, 12449, 12450, 12451, 12452, 12453, 12454, 12455, 
              12456, 12457, 12458, 12459, 12460, 12461, 12462, 12463, 12464, 
              12465, 12466, 12467, 12468, 12469, 12470, 12471, 12472, 12473, 
              12474, 12475, 12476, 12477, 12478, 12479, 12480, 12481, 12482, 
              12483, 12484, 12485, 12486, 12487, 12488, 12489, 12490, 12491, 
              12492, 12493, 12494, 12495, 12496, 12497, 12498, 12499, 12500, 
              12501, 12502, 12503, 12504, 12505, 12506, 12507, 12508, 12509, 
              12510, 12511, 12512, 12513, 12514, 12515, 12516, 12517, 12518, 
              12519, 12520, 12521, 12522, 12523, 12524, 12525, 12526, 12527, 
              12528, 12529, 12530, 12531, 12532, 12533, 12534, 12535, 12536, 
              12537, 12538, 12539, 12540, 12541, 12542, 12543, 12544, 12545, 
              12546, 12547, 12548, 12549, 12550, 12551, 12552, 12553, 12554, 
              12555, 12556, 12557, 12558, 12559, 12560, 12561, 12562, 12563, 
              12564, 12565, 12566, 12567, 12568, 12569, 12570, 12571, 12572, 
              12573, 12574, 12575, 12576, 12577, 12578, 12579, 12580, 12581, 
              12582, 12583, 12584, 12585, 12586, 12587, 12588, 12589, 12590, 
              12591, 12592, 12593, ]
    add_fv_list = [[3316, 3318, 3315], [3318, 3313, 3315],
                   [3313, 3310, 3315], [3313, 3304, 3310],
                   [3304, 3307, 3310], [3303, 3307, 3304],
                   [3303, 3300, 3307], [3291, 3300, 3303],
                   [3291, 3296, 3300], [3292, 3297, 3296],
                   [3292, 3296, 3291], [3292, 3294, 3297],
                   [6718, 6715, 6716], [6713, 6718, 6716],
                   [6713, 6716, 6711], [6704, 6713, 6711],
                   [6704, 6711, 6707], [6703, 6704, 6707],
                   [6703, 6707, 6701], [6692, 6703, 6701],
                   [6692, 6701, 6696], [6691, 6692, 6696],
                   [6691, 6696, 6697], [6694, 6691, 6697]]
    face_list = mesh.face_vertex_indices().tolist()
    new_face_list = []
    for i in range(len(face_list)):
        if not i in d_inds:
            new_face_list.append(face_list[i])
    new_face_list = new_face_list + add_fv_list
    new_mesh = make_trimesh(mesh.points(), np.array(new_face_list))
    return new_mesh

# flatten naval in smpl mesh 
def flatten_naval(mesh):
    verts = mesh.points()
    verts[5234] = (verts[4402]+verts[3504])*0.5
    verts[1767] = (verts[3504]+verts[917])*0.5
    verts[1337] = (verts[917]+verts[1769])*0.5
    verts[4813] = (verts[1769]+verts[4402])*0.5
    verts[4812] = (verts[5234]+verts[4813])*0.5
    verts[3501] = (verts[5234]+verts[1767])*0.5
    verts[1336] = (verts[1767]+verts[1337])*0.5
    verts[1768] = (verts[1337]+verts[4813])*0.5
    verts[3500] = (verts[3504]+verts[4402]+verts[917]+verts[1769])*0.25
    return mesh

# rotate verts along y axis
def rotate_verts_y(verts, y):
    verts_mean = np.mean(verts, axis = 0)
    verts = verts - verts_mean

    angle = y*math.pi/180
    R = np.array([[np.cos(angle), 0, np.sin(angle)],
                  [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])

    for i in range(len(verts)):
        verts[i] = np.dot(R, verts[i])
    verts = verts + verts_mean
    return verts

# rotate verts along x axis
def rotate_verts_x(verts, x):
    verts_mean = np.mean(verts, axis = 0)
    verts = verts - verts_mean

    angle = x*math.pi/180
    R = np.array([[1, 0, 0],
                  [0, np.cos(angle), -np.sin(angle)],
                  [0, np.sin(angle), np.cos(angle)]])

    for i in range(len(verts)):
        verts[i] = np.dot(R, verts[i])
    verts = verts + verts_mean
    return verts

# rotate verts along z axis
def rotate_verts_z(verts, z):
    verts_mean = np.mean(verts, axis = 0)
    verts = verts - verts_mean

    angle = z*math.pi/180
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])

    for i in range(len(verts)):
        verts[i] = np.dot(R, verts[i])
    verts = verts + verts_mean
    return verts

# used in argument parser
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
