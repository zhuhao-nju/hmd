from __future__ import print_function
import os
import torch
import torch.nn.functional as F
import numpy as np
import math 


#for single depth [stable version]
def depthToNormal(depthmap,
                  device,
                  K,
                  thres,
                  img_size
                 ):
    # default
    K_torch = torch.tensor([K[0], K[1], K[2], K[3]]).to(device)

    C, H, W = depthmap.size()
    assert( C==1 and H == img_size[0] and W == img_size[1])
    
    depthmap = torch.reshape(depthmap, [H,W]) #resize to H W 
    
    X_grid, Y_grid = torch.meshgrid( [torch.arange(H, out=torch.FloatTensor().to(device)), torch.arange(W, out=torch.FloatTensor().to(device))] )
    
    X = (X_grid - K_torch[2]) * depthmap / K_torch[0] 
    Y = (Y_grid - K_torch[3]) * depthmap / K_torch[1]
    
    DepthPoints = torch.stack([X, Y, depthmap], dim=2) # all 3D point 
    
    delta_right = DepthPoints[2:,1:-1,:] - DepthPoints[1:-1,1:-1,:]
    delta_down  = DepthPoints[1:-1,2:,:] - DepthPoints[1:-1,1:-1,:]
    
    delta_left = DepthPoints[0:-2,1:-1,:] - DepthPoints[1:-1,1:-1,:]
    delta_up   = DepthPoints[1:-1,0:-2,:] - DepthPoints[1:-1,1:-1,:]
    
    
    normal_crop1 = torch.cross(delta_down, delta_right)
    normal_crop1 = F.normalize(normal_crop1, p=2, dim=2)
    
    normal_crop2 = torch.cross(delta_up,   delta_left )
    normal_crop2 = F.normalize(normal_crop2, p=2, dim=2)
    
    normal_crop  = normal_crop1 + normal_crop2
    normal_crop  = F.normalize(normal_crop, p=2, dim=2)
    
    
    normal = torch.zeros(H,W,3).to(device)
    normal[1:-1,1:-1,:] = normal_crop
    
    confidence_map_crop = torch.ones(H-2,W-2).to(device)
    
    delta_right_norm = torch.norm(delta_right, p=2, dim=2)
    delta_down_norm  = torch.norm(delta_down, p=2, dim=2)
    confidence_map_crop[ delta_right_norm > thres] = 0.0
    confidence_map_crop[ delta_down_norm > thres] = 0.0
    
    confidence_map = torch.zeros(H,W)
    confidence_map[1:-1,1:-1] = confidence_map_crop
    confidence_map[depthmap == 0] = 0
    
    # change to CHW
    normal = normal.permute(2, 0, 1)
    
    # return the  normal [3,H,W] and confidence map
    return normal, confidence_map
  
    
def normalToSH(normal, 
               device):
    
    #here is the SH (SH Basis order=2)
    c =torch.zeros(9).to(device)
    
    c[0] =  1/(2* math.sqrt(math.pi) );
    c[1] = - math.sqrt(3) / ( 2* math.sqrt(math.pi) )
    c[2] = - math.sqrt(3) / ( 2* math.sqrt(math.pi) )
    c[3] =  math.sqrt(3) / ( 2* math.sqrt(math.pi) )
    c[4] =  math.sqrt(15) / ( 2* math.sqrt(math.pi) )
    c[5] = - math.sqrt(15) / ( 2* math.sqrt(math.pi) ) 
    c[6] = - math.sqrt(15) / ( 2* math.sqrt(math.pi) ) 
    c[7] =  math.sqrt(15) / ( 4* math.sqrt(math.pi) )
    c[8] =  math.sqrt(5) / ( 4* math.sqrt(math.pi) )    
    
    C, H, W = normal.size()
    
    spherical_harmonics = torch.zeros(9,H,W).to(device)
    
    spherical_harmonics[0,:,:] = 1 * c[0];  
    spherical_harmonics[1,:,:] = normal[0,:,:] * c[1];
    spherical_harmonics[2,:,:] = normal[1,:,:] * c[2];
    spherical_harmonics[3,:,:] = normal[2,:,:] * c[3];
    spherical_harmonics[4,:,:] = normal[0,:,:] * normal[1,:,:]  * c[4];
    spherical_harmonics[5,:,:] = normal[0,:,:] * normal[2,:,:]  * c[5];
    spherical_harmonics[6,:,:] = normal[1,:,:] * normal[2,:,:]  * c[6];
    spherical_harmonics[7,:,:] = (normal[0,:,:] * normal[0,:,:] -  normal[1,:,:] * normal[1,:,:]) * c[7];
    spherical_harmonics[8,:,:] = (3 * normal[2,:,:] * normal[2,:,:] - 1) *c[8];
    
    return spherical_harmonics


def RGBalbedoSHToLight(colorImg, albedoImg, SH, confidence_map):
    
    #remove non-zeros [now confidence_map is the more clean] 
    confidence_map[colorImg==0] = 0
    confidence_map[albedoImg==0] = 0
    
    id_non_not = confidence_map.nonzero()
    idx_non = torch.unbind(id_non_not, 1) # this only works for two dimesion
    
    colorImg_non = colorImg[idx_non]
    albedoImg_non = albedoImg[idx_non]
    
    #get the shadingImg element-wise divide
    shadingImg_non = torch.div(colorImg_non, albedoImg_non)    
    shadingImg_non2 = shadingImg_non.view(-1,1)

    #:means 9 channels [get the shading image]
    SH0 = SH[0,:,:]; SH0_non = SH0[idx_non]
    SH1 = SH[1,:,:]; SH1_non = SH1[idx_non]
    SH2 = SH[2,:,:]; SH2_non = SH2[idx_non]
    SH3 = SH[3,:,:]; SH3_non = SH3[idx_non]
    SH4 = SH[4,:,:]; SH4_non = SH4[idx_non]
    SH5 = SH[5,:,:]; SH5_non = SH5[idx_non]
    SH6 = SH[6,:,:]; SH6_non = SH6[idx_non]
    SH7 = SH[7,:,:]; SH7_non = SH7[idx_non]
    SH8 = SH[8,:,:]; SH8_non = SH8[idx_non]
           
    SH_NON = torch.stack([SH0_non, SH1_non, SH2_non, SH3_non, SH4_non, SH5_non, SH6_non, SH7_non, SH8_non], dim=-1)
    
    ## only use the first N soultions if M>N  A(M*N) B(N*K) X should (N*K)[use N if M appears] 
    ## torch.gels(B, A, out=None)  Tensor
    ## https://pytorch.org/docs/stable/torch.html#torch.gels  
    light, _ = torch.gels(shadingImg_non2, SH_NON)  
    light_9 = light[0:9] # use first 9

    return (light_9, SH)


def RGBDalbedoToLight(colorImg, 
                      depthImg, 
                      albedoImg, 
                      device,
                      K = [400.0, 400.0, 224.0, 224.0],
                      thres = 30, 
                      img_size= [448,448]):
    
    normal, confidence_map = depthToNormal(depthImg, device, K, thres, img_size)
    SH = normalToSH(normal, device)
    lighting_est = RGBalbedoSHToLight(colorImg, albedoImg, SH, confidence_map)
    
    return lighting_est


#for Batch depth to Normal [stable]
def depthToNormalBatch(depthmap,
                  device,
                  K = [400.0, 400.0, 224.0, 224.0],
                  thres = 30,
                  img_size =[448, 448]):
    
    # default
    K_torch = torch.tensor([K[0], K[1], K[2], K[3]]).to(device)

    N, C, H, W = depthmap.size()
    assert( C==1 and H == img_size[0] and W == img_size[1])
    
    depthmap = torch.reshape(depthmap, [N,H,W]) #resize to (N H W) 
    
    X_grid, Y_grid = torch.meshgrid( [torch.arange(H, out=torch.FloatTensor().to(device)), torch.arange(W, out=torch.FloatTensor().to(device))] )
    X_grid = X_grid.repeat(N, 1, 1) # repeat to N H W 
    Y_grid = Y_grid.repeat(N, 1, 1) # repeat to N H W
    
    
    X = (X_grid - K_torch[2]) *depthmap / K_torch[0]
    Y = (Y_grid - K_torch[3]) *depthmap / K_torch[1]
    
    DepthPoints = torch.stack([X, Y, depthmap], dim=3) # all 3D point 
    
    delta_right = DepthPoints[:, 2:,1:-1,:] - DepthPoints[:, 1:-1,1:-1,:]
    delta_down  = DepthPoints[:, 1:-1,2:,:] - DepthPoints[:, 1:-1,1:-1,:]
    
    delta_left = DepthPoints[:, 0:-2,1:-1,:] - DepthPoints[:, 1:-1,1:-1,:]
    delta_up   = DepthPoints[:, 1:-1,0:-2,:] - DepthPoints[:, 1:-1,1:-1,:]
    
    
    normal_crop1 = torch.cross(delta_down, delta_right)
    normal_crop1 = F.normalize(normal_crop1, p=2, dim=3)
    
    normal_crop2 = torch.cross(delta_up,   delta_left )
    normal_crop2 = F.normalize(normal_crop2, p=2, dim=3)
    
    normal_crop  = normal_crop1 #+ normal_crop2
    normal_crop  = F.normalize(normal_crop, p=2, dim=3)
    
    
#     normal_crop = torch.cross(delta_down, delta_right)
    normal = torch.zeros(N,H,W,3).to(device)
    normal[:, 1:-1, 1:-1, :] = normal_crop
    
    confidence_map_crop = torch.ones(N, H-2, W-2).to(device)
    
    delta_right_norm = torch.norm(delta_right, p=2, dim=3)
    delta_down_norm = torch.norm(delta_down, p=2, dim=3)
    confidence_map_crop[ delta_right_norm > thres ] =0.0
    confidence_map_crop[ delta_down_norm > thres ] =0.0
    
    delta_left_norm = torch.norm(delta_left, p=2, dim=3)
    delta_up_norm = torch.norm(delta_up, p=2, dim=3)
    confidence_map_crop[ delta_left_norm > thres ] =0.0
    confidence_map_crop[ delta_up_norm > thres ] =0.0
    
    confidence_map = torch.zeros(N,H,W)
    confidence_map[:, 1:-1, 1:-1] = confidence_map_crop
    confidence_map[depthmap == 0] = 0 # [N, H, W]
    
   
    # change to CHW    
    normal = normal.permute(0, 3, 1, 2)  # return the  normal [N,C,H,W] and confidence map
   
    return normal, confidence_map
    #return normal

    
def normalToSHBatch(normal, 
               device):
    
    N, CC, H ,W= normal.size()
    #here is the SH (SH Basis order=2)
    c =torch.zeros(9).to(device)
    
    c[0] =  1/(2* math.sqrt(math.pi) );
    c[1] = - math.sqrt(3) / ( 2* math.sqrt(math.pi) )
    c[2] = - math.sqrt(3) / ( 2* math.sqrt(math.pi) )
    c[3] =  math.sqrt(3) / ( 2* math.sqrt(math.pi) )
    c[4] =  math.sqrt(15) / ( 2* math.sqrt(math.pi) )
    c[5] = - math.sqrt(15) / ( 2* math.sqrt(math.pi) ) 
    c[6] = - math.sqrt(15) / ( 2* math.sqrt(math.pi) ) 
    c[7] =  math.sqrt(15) / ( 4* math.sqrt(math.pi) )
    c[8] =  math.sqrt(5) / ( 4* math.sqrt(math.pi) )    
    
    
    spherical_harmonics = torch.zeros(N,H,W,9).to(device)
    
    spherical_harmonics[:,:,:,0] = 1 * c[0];  
    spherical_harmonics[:,:,:,1] = normal[:,0,:,:] * c[1];
    spherical_harmonics[:,:,:,2] = normal[:,1,:,:] * c[2];
    spherical_harmonics[:,:,:,3] = normal[:,2,:,:] * c[3];
    spherical_harmonics[:,:,:,4] = normal[:,0,:,:] * normal[:,1,:,:]  * c[4];
    spherical_harmonics[:,:,:,5] = normal[:,0,:,:] * normal[:,2,:,:]  * c[5];
    spherical_harmonics[:,:,:,6] = normal[:,1,:,:] * normal[:,2,:,:]  * c[6];
    spherical_harmonics[:,:,:,7] = (normal[:,0,:,:] * normal[:,0,:,:] -  normal[:,1,:,:] * normal[:,1,:,:]) * c[7];
    spherical_harmonics[:,:,:,8] = (3 * normal[:,2,:,:] * normal[:,2,:,:] - 1) *c[8];
    
    return spherical_harmonics
