from __future__ import print_function 
import numpy as np
import PIL.Image
import openmesh as om
import argparse
import os
import sys
sys.path.append("../src/")
from renderer import SMPLRenderer
from hmr_predictor import hmr_predictor
from hmr_predictor import proc_sil
from hmr_predictor import preproc_img
from utility import make_trimesh
from utility import pad_arr

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--img', required = True, 
                    help = 'input image file')
parser.add_argument('--outf', default = '', 
                    help = 'output folder (str, default is image name)')
parser.add_argument('--pad', type = int, default = 0,
                    help = 'padding')
parser.add_argument('--crop_x', nargs = '+', type = int, default = -1,
                    help = 'bounding box in x direction, to edge (2 integers)')
parser.add_argument('--crop_y', nargs = '+', type = int, default = -1,
                    help = 'bounding box in y direction, to edge (2 integers)')
opt = parser.parse_args()

if opt.outf == '':
    opt.outf = "./result/" + opt.img.split("/")[-1][:-4] + "/"
print(opt)

src_img = np.array(PIL.Image.open(opt.img))
crop_img = src_img.copy()

if opt.crop_x != -1:
    if len(opt.crop_x) != 2:
        print("ERROR: crop_x must be a list with 2 elements")
    crop_img = crop_img[:, opt.crop_x[0]:-opt.crop_x[1]]
if opt.crop_y != -1:
    if len(opt.crop_y) != 2:
        print("ERROR: crop_y must be a list with 2 elements")
    crop_img = crop_img[opt.crop_y[0]:-opt.crop_y[1], :]
if opt.pad>0:
    crop_img = pad_arr(crop_img, 50)
std_img, proc_para = preproc_img(crop_img, img_size = 224, 
                                 margin = 30, normalize = True)

# initialize
hmr_pred = hmr_predictor()
renderer = SMPLRenderer()
faces = np.load("../predef/smpl_faces.npy")

# hmr predict
verts, cam, proc_para, std_img = hmr_pred.predict(std_img, normalize=False)

# build output folder if not exist
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# write results
result_mesh = make_trimesh(verts, faces)
om.write_mesh(opt.outf + "hmr_mesh.obj", result_mesh)

final_img = ((std_img.copy()+1)*127).astype(np.uint8)
PIL.Image.fromarray(final_img).save(opt.outf + "std_img.jpg")

print("%s - finished, results are saved to [%s]" % (opt.img, opt.outf))
print("hmr done")
