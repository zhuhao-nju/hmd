import numpy as np
import openmesh as om
from tqdm import trange
import sys, os, pickle, argparse, PIL.Image
sys.path.append("../src/")
from hmr_predictor import hmr_predictor
from utility import make_trimesh

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
output_dir = "./eval_data/%s_set/pred_save/" % opt.set
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

hmr_pred = hmr_predictor()
faces = np.load("../predef/smpl_faces.npy")

tr = trange(data_num, desc='Bar desc', leave=True)
for i in tr:
    src_img = np.array(PIL.Image.open("./eval_data/%s_set/input/%03d_img.png" % \
                                      (opt.set, i)))[:,:,:3]
    verts, cam, proc_para, std_img = hmr_pred.predict(src_img)
    
    with open(output_dir + "pre_%03d.pkl" % i, 'wb') as fp:
        pickle.dump({"verts": verts, 
                     "cam": cam, 
                     "proc_para": proc_para, 
                     "std_img": std_img, 
                    }, fp)
    
    mesh = make_trimesh(verts, faces)
    om.write_mesh("./eval_data/%s_set/pred_save/hmr_%03d.obj" % (opt.set, i), mesh)
