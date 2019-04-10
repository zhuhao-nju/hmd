import numpy as np
import scipy.ndimage

# select from LSPET dataset
# RULE: TESTED
# (1) valid point number >=12
# (2) all joints are inside image (ignore invalid joints)
# (3) all joints are inside silhouette (sils are morphology dilated)
def lsp_filter(joints, sil):
    # make sil one channel
    if len(sil.shape) == 3:
        sil = sil[:, :, 0]
    sil = scipy.ndimage.morphology.binary_dilation(sil, iterations = 2)
    
    invalid_num = np.sum(joints[2,:])
    if invalid_num > 5:
        #print("filtered due to too few valid points")
        return False
    
    for i in range(14):
        if joints[2,i] == 0:
            continue
        x = int(joints[0,i])
        y = int(joints[1,i])
        if x>=sil.shape[1] or y>=sil.shape[0] or x<0 or y<0:
            #print("filtered due to outside image")
            return False
        if sil[y, x] == 0:
            #print("filtered due to outside sil")
            return False   
    return True

# select from LSPET dataset
# RULE: TESTED
# (1) valid point number >=12
# (2) all joints are inside image (ignore invalid joints)
# (3) all joints are inside silhouette (sils are morphology dilated)
def lspet_filter(joints, sil):
    # make sil one channel
    if len(sil.shape) == 3:
        sil = sil[:, :, 0]
    # to wipe out the region out side the silhouette but occur due to the 
    # the existence of joint point, refer to #00007 in LSPET dataset
    sil = scipy.ndimage.morphology.binary_erosion(sil, iterations = 1)
    sil = scipy.ndimage.morphology.binary_dilation(sil, iterations = 2)
    
    valid_num = np.sum(joints[2,:])
    if valid_num < 12:
        #print("filtered due to too few valid points")
        return False
    
    for i in range(14):
        if joints[2,i] == 0:
            continue
        x = int(joints[0,i])
        y = int(joints[1,i])
        if x>=sil.shape[1] or y>=sil.shape[0] or x<0 or y<0:
            #print("filtered due to outside image")
            return False
        if sil[y, x] == 0:
            #print("filtered due to outside sil")
            return False
    return True


# select from MPII dataset
# RULE: TESTED
# (1) all joints are inside image
# (2) all joints are inside sil (sils are morphology dilated, for head joint)
def mpii_filter(joints, sil):
    # make sil one channel
    if len(sil.shape) == 3:
        sil = sil[:, :, 0]
    sil = scipy.ndimage.morphology.binary_dilation(sil, iterations = 2)
    
    for i in range(14):
        x = int(joints[0,i])
        y = int(joints[1,i])
        if x>=sil.shape[1] or y>=sil.shape[0] or x<0 or y<0:
            #print("filtered due to outside image")
            return False
        if sil[y, x] == 0:
            #print("filtered due to outside sil")
            return False
    return True


# select from COCO dataset
# RULE: TESTED
# (1) object num = 1; # this is done before this function run
# (2) silhouetee seg_points number > 100 # this is done before this function run
# (3) non-zero key points >=14;
# (4) all key points are inside image
# (5) all key points are in silhouette
def coco_filter(key_points, sil):

    # count key points number
    kp_num = 17
    for i in range(17):
        if np.array_equal(key_points[i], np.array([0, 0, 0])):
            kp_num -= 1

    # filter if kp is too few
    if kp_num < 12:
        #print("filter out by too few key points")
        return False

    # check if all kps are in image and sil
    for i in range(len(key_points)):
        if i < 5:
            # head points always outside the sil
            continue
        x = int(key_points[i,0])
        y = int(key_points[i,1])
        if x>=sil.shape[1] or y>=sil.shape[0] or x<0 or y<0:
            #print("filtered due to outside image")
            return False
        if sil[y, x] == 0:
            #print("filtered due to outside sil")
            return False

    return True
