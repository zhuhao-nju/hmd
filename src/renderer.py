"""
Renders mesh using OpenDr for visualization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.renderer import DepthRenderer
from opendr.lighting import LambertianPointLight

colors = {
    'light_blue': [0.65098039, 0.74117647, 0.85882353], # ID = 0, light blue
    'light_black': [.0, .0, .0], # ID = 1, black, for mask generate
    'light_white': [1.0, 1.0, 1.0,], # ID = 2, white
    'light_pink': [.9, .7, .7] # ID = 3, pink
}

# error print func
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class SMPLRenderer(object):
    def __init__(self,
                 img_size=224,
                 flength=500.,
                 face_path="../predef/smpl_faces.npy",
                 verts = None,
                 vert_colors = None,
                 cam = None):
        self.faces = np.load(face_path)
        self.verts = verts
        self.vert_colors = vert_colors
        self.cam = cam
        self.w = img_size
        self.h = img_size
        self.flength = flength

    def __call__(self,
                 verts=None,
                 faces=None,
                 vert_colors=None,
                 cam=None,
                 img=None,
                 do_alpha=False,
                 far=None,
                 near=None,
                 color_id=0,
                 img_size=None):
        """
        cam is 3D [f, px, py]
        """
        if verts is None:
            if self.verts is not None:
                verts = self.verts
            else:
                eprint("No verts found, use argument verts "
                       "or set_verts function")
                return False
        if faces is None:
            faces = self.faces
        if vert_colors is None and self.vert_colors is not None:
            vert_colors = self.vert_colors
                
        if img is not None:
            h, w = img.shape[:2]
        elif img_size is not None:
            h = img_size[0]
            w = img_size[1]
        else:
            h = self.h
            w = self.w

        if cam is None:
            if self.cam is not None:
                cam = self.cam
            else:
                cam = [self.flength, w / 2., h / 2.]

        use_cam = ProjectPoints(
            f=cam[0] * np.ones(2),
            rt=np.zeros(3),
            t=np.zeros(3),
            k=np.zeros(5),
            c=cam[1:3])

        if near is None:
            near = np.maximum(np.min(verts[:, 2]) - 25, 0.1)
        if far is None:
            far = np.maximum(np.max(verts[:, 2]) + 25, 25)
    
        imtmp = render_model(
            verts,
            faces,
            w,
            h,
            use_cam,
            vert_colors = vert_colors,
            do_alpha = do_alpha,
            img = img,
            far = far,
            near = near,
            color_id = color_id)

        return (imtmp * 255).astype('uint8')

    def set_verts(self, verts):
        self.verts = verts.copy()
        return 
    
    def rotated(self,
                verts,
                deg,
                cam=None,
                axis='y',
                img=None,
                do_alpha=True,
                far=None,
                near=None,
                color_id=0,
                img_size=None):
        import math
        if axis == 'y':
            around = cv2.Rodrigues(np.array([0, math.radians(deg), 0]))[0]
        elif axis == 'x':
            around = cv2.Rodrigues(np.array([math.radians(deg), 0, 0]))[0]
        else:
            around = cv2.Rodrigues(np.array([0, 0, math.radians(deg)]))[0]
        center = verts.mean(axis=0)
        new_v = np.dot((verts - center), around) + center

        return self.__call__(
            new_v,
            cam,
            img=img,
            do_alpha=do_alpha,
            far=far,
            near=near,
            img_size=img_size,
            color_id=color_id)
    
    def silhouette(self,
                   norm=True,
                   verts=None,
                   faces=None,
                   cam=None,
                   do_alpha=False,
                   far=None,
                   near=None,
                   img_size=None):
        sil = self.__call__(verts=verts,
                            faces=faces,
                            cam=cam,
                            img=None, # use default white as background
                            do_alpha=False,
                            far=far,
                            near=near,
                            color_id=1, # black foreground
                            img_size=img_size)
        sil = np.amax(sil, axis=2)
        foreground = sil<128
        background = sil>=128
        if norm is True:
            sil[foreground] = 1
            sil[background] = 0
        else:
            sil[foreground] = 255
            sil[background] = 0
        return sil

def _create_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.5,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn


def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                   [-np.sin(angle), 0., np.cos(angle)]])
    return np.dot(points, ry)


def simple_renderer(rn,
                    verts,
                    faces,
                    vert_colors = None,
                    yrot = np.radians(120),
                    color = colors['light_pink']):
    # Rendered model color
    if vert_colors is None:
        rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))
    else:
        rn.set(v=verts, f=faces, vc=vert_colors, bgcolor=np.ones(3))
    albedo = rn.vc
    
    
    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))
    
    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, -200, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))
    
    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, -200, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))

    return rn.r


def get_alpha(imtmp, bgval=1.):
    h, w = imtmp.shape[:2]
    alpha = (~np.all(imtmp == bgval, axis=2)).astype(imtmp.dtype)

    b_channel, g_channel, r_channel = cv2.split(imtmp)

    im_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha.astype(
        imtmp.dtype)))
    return im_RGBA


def append_alpha(imtmp):
    alpha = np.ones_like(imtmp[:, :, 0]).astype(imtmp.dtype)
    if np.issubdtype(imtmp.dtype, np.uint8):
        alpha = alpha * 255
    b_channel, g_channel, r_channel = cv2.split(imtmp)
    im_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha))
    return im_RGBA


def render_model(verts,
                 faces,
                 w,
                 h,
                 cam,
                 vert_colors = None,
                 near = 0.5,
                 far = 25,
                 img = None,
                 do_alpha = False,
                 color_id = None,
                ):
    rn = _create_renderer(
        w=w, h=h, near=near, far=far, rt=cam.rt, t=cam.t, f=cam.f, c=cam.c)

    # Uses img as background, otherwise white background.
    if img is not None:
        rn.background_image = img / 255. if img.max() > 1 else img

    if color_id is None:
        color = colors['light_blue']
    else:
        color_list = colors.values()
        color = color_list[color_id % len(color_list)]

    imtmp = simple_renderer(rn, 
                            verts, 
                            faces, 
                            vert_colors = vert_colors, 
                            color = color, 
                           )

    # If white bg, make transparent.
    if img is None and do_alpha:
        imtmp = get_alpha(imtmp)
    elif img is not None and do_alpha:
        imtmp = append_alpha(imtmp)

    return imtmp


# render mesh to depth map using OpenDR
def render_depth(mesh, require_visi = False, img_size = (448, 448), f = 1000):
    rn = DepthRenderer()
    rn.camera = ProjectPoints(rt = np.zeros(3), 
                              t = np.zeros(3), 
                              f = np.array([f, f]),
                              c = np.array([img_size[1], img_size[0]])/2.,
                              k = np.zeros(5)
                             )
    rn.frustum = {'near': .5, 'far': 10., 
                  'width': img_size[1], 'height': img_size[0]}
    rn.v = mesh.points()
    rn.f = mesh.face_vertex_indices()
    rn.bgcolor = np.zeros(3)
    if require_visi is True:
        return rn.r, rn.visibility_image
    else:
        return rn.r
    
# render verts to depth map using OpenDR
def render_depth_v(verts, faces, 
                   require_visi = False, img_size=(448, 448), f=1000):
    rn = DepthRenderer()
    rn.camera = ProjectPoints(rt = np.zeros(3), 
                              t = np.zeros(3), 
                              f = np.array([f, f]),
                              c = np.array([img_size[1], img_size[0]])/2.,
                              k = np.zeros(5)
                             )
    rn.frustum = {'near': .5, 'far': 10., 
                  'width': img_size[1], 'height': img_size[0]}
    rn.v = verts
    rn.f = faces
    rn.bgcolor = np.zeros(3)
    if require_visi is True:
        return rn.r, rn.visibility_image
    else:
        return rn.r
    
# render mesh to boundary image using OpenDR
def render_bound(mesh, require_id = False, img_size=(448, 448), f=1000):
    rn = DepthRenderer()
    rn.camera = ProjectPoints(rt = np.zeros(3), 
                              t = np.zeros(3), 
                              f = np.array([f, f]),
                              c = np.array([img_size[1], img_size[0]])/2.,
                              k = np.zeros(5)
                             )
    rn.frustum = {'near': .5, 'far': 10., 
                  'width': img_size[1], 'height': img_size[0]}
    rn.v = mesh.points()
    rn.f = mesh.face_vertex_indices()
    rn.bgcolor = np.zeros(3)
    if require_id is False:
        return rn.boundarybool_image
    else:
        return rn.boundaryid_image
    
    
# render mesh to silhouette using OpenDR
def render_sil(mesh, norm = True, img_size=(448, 448), f=1000):
    rn = DepthRenderer()
    rn.camera = ProjectPoints(rt = np.zeros(3), 
                              t = np.zeros(3), 
                              f = np.array([f, f]),
                              c = np.array([img_size[1], img_size[0]])/2.,
                              k = np.zeros(5)
                             )
    rn.frustum = {'near': .5, 'far': 10., 
                  'width': img_size[1], 'height': img_size[0]}
    rn.v = mesh.points()
    rn.f = mesh.face_vertex_indices()
    rn.bgcolor = np.zeros(3)
    depth_map = rn.r
    sil_img = np.zeros(depth_map.shape, dtype = np.uint8)
    if norm is True:
        sil_img[depth_map<10] = 1
    else:
        sil_img[depth_map<10] = 255
    return sil_img



class all_renderer(object):
    def __init__(self,
                 img_size = 224,
                 flength = 500.,
                 cam = None):
        self.cam = cam
        self.w = img_size
        self.h = img_size
        self.flength = flength

    def __call__(self,
                 verts,
                 faces,
                 vert_colors=None,
                 cam=None,
                 img=None,
                 do_alpha=False,
                 far=None,
                 near=None,
                 color_id=0,
                 img_size=None):
        """
        cam is 3D [f, px, py]
        """
                
        if img is not None:
            h, w = img.shape[:2]
        elif img_size is not None:
            h = img_size[0]
            w = img_size[1]
        else:
            h = self.h
            w = self.w

        if cam is None:
            if self.cam is not None:
                cam = self.cam
            else:
                cam = [self.flength, w / 2., h / 2.]

        use_cam = ProjectPoints(
            f=cam[0] * np.ones(2),
            rt=np.zeros(3),
            t=np.zeros(3),
            k=np.zeros(5),
            c=cam[1:3])

        if near is None:
            near = np.maximum(np.min(verts[:, 2]) - 25, 0.1)
        if far is None:
            far = np.maximum(np.max(verts[:, 2]) + 25, 25)
    
        imtmp = render_model(
            verts,
            faces,
            w,
            h,
            use_cam,
            vert_colors = vert_colors,
            do_alpha = do_alpha,
            img = img,
            far = far,
            near = near,
            color_id = color_id)

        return (imtmp * 255).astype('uint8')

    def set_verts(self, verts):
        self.verts = verts.copy()
        return 
    
    def rotated(self,
                verts,
                deg,
                cam=None,
                axis='y',
                img=None,
                do_alpha=True,
                far=None,
                near=None,
                color_id=0,
                img_size=None):
        import math
        if axis == 'y':
            around = cv2.Rodrigues(np.array([0, math.radians(deg), 0]))[0]
        elif axis == 'x':
            around = cv2.Rodrigues(np.array([math.radians(deg), 0, 0]))[0]
        else:
            around = cv2.Rodrigues(np.array([0, 0, math.radians(deg)]))[0]
        center = verts.mean(axis=0)
        new_v = np.dot((verts - center), around) + center

        return self.__call__(
            new_v,
            cam,
            img=img,
            do_alpha=do_alpha,
            far=far,
            near=near,
            img_size=img_size,
            color_id=color_id)
    
    def silhouette(self,
                   norm=True,
                   verts=None,
                   faces=None,
                   cam=None,
                   do_alpha=False,
                   far=None,
                   near=None,
                   img_size=None):
        sil = self.__call__(verts=verts,
                            faces=faces,
                            cam=cam,
                            img=None, # use default white as background
                            do_alpha=False,
                            far=far,
                            near=near,
                            color_id=1, # black foreground
                            img_size=img_size)
        sil = np.amax(sil, axis=2)
        foreground = sil<128
        background = sil>=128
        if norm is True:
            sil[foreground] = 1
            sil[background] = 0
        else:
            sil[foreground] = 255
            sil[background] = 0
        return sil
