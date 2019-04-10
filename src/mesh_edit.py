import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, cg, eigsh
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle
import sparseqr
import time

WEIGHT = 1.0

##############################################################
##                  Laplacian Mesh Editing                  ##
##############################################################

#Purpose: To return a sparse matrix representing a Laplacian matrix with
#the graph Laplacian (D - A) in the upper square part and anchors as the
#lower rows
#Inputs: mesh (polygon mesh object), anchorsIdx (indices of the anchor points)
#Returns: L (An (N+K) x N sparse matrix, where N is the number of vertices
#and K is the number of anchors)
def getLaplacianMatrixUmbrella(mesh, anchorsIdx):
    n = mesh.n_vertices() # N x 3
    k = anchorsIdx.shape[0]
    I = []
    J = []
    V = []
    
    vv_idx_list = list(mesh.vertex_vertex_indices())
    # Build sparse Laplacian Matrix coordinates and values
    for i in range(n):
        idx_nbr = filter(lambda x:x != -1, vv_idx_list[i])
        num_nbr = len(idx_nbr)
        
        I = I + ([i] * (num_nbr + 1)) # repeated row
        J = J + idx_nbr + [i] # column indices and this row
        V = V + ([-1] * num_nbr) + [num_nbr] # negative weights and row degree

    # augment Laplacian matrix with anchor weights  
    for i in range(k):
        I = I + [n + i]
        J = J + [anchorsIdx[i]]
        V = V + [WEIGHT] # default anchor weight
    
    L = sparse.coo_matrix((V, (I, J)), shape=(n + k, n)).tocsr()
    return L

# Modified for openmesh.mesh, Note that only suitable for watertight model
#Purpose: To return a sparse matrix representing a laplacian matrix with
#cotangent weights in the upper square part and anchors as the lower rows
#Inputs: mesh (polygon mesh object), anchorsIdx (indices of the anchor points)
#Returns: L (An (N+K) x N sparse matrix, where N is the number of vertices
#and K is the number of anchors)
def getLaplacianMatrixCotangent(mesh, anchorsIdx):
    n = mesh.n_vertices() # N x 3
    k = anchorsIdx.shape[0]
    I = []
    J = []
    V = []
    #l = mesh.vertex_vertex_indices()
    for v in mesh.vertices():
        weights = []
        p_this = mesh.point(v)
        p_nbrs = []
        id_this = v.idx()
        id_nbrs = []
        for vv in mesh.vv(v):
            p_nbrs.append(mesh.point(vv))
            id_nbrs.append(vv.idx())
        num_nbr = len(id_nbrs)
        for i in range(num_nbr):
            u = p_this - p_nbrs[(i+num_nbr-1)%num_nbr]
            v = p_nbrs[(i+num_nbr)%num_nbr]- p_nbrs[(i+num_nbr-1)%num_nbr]
            cotangent_1 = (np.dot(u, v)
                          /np.sqrt(np.sum(np.square(np.cross(u, v)))))
            u = p_this - p_nbrs[(i+num_nbr+1)%num_nbr]
            v = p_nbrs[(i+num_nbr)%num_nbr]- p_nbrs[(i+num_nbr+1)%num_nbr]
            cotangent_2 = (np.dot(u, v)
                          /np.sqrt(np.sum(np.square(np.cross(u, v)))))
            weights.append(-0.5 * (cotangent_1 + cotangent_2)) # cotangent weights

        I = I + ([id_this] * (num_nbr + 1)) # repeated row
        J = J + id_nbrs + [id_this] # column indices and this row
        V = V + weights + [(-1 * np.sum(weights))] # n negative weights and row vertex sum    

    # augment Laplacian matrix with anchor weights  
    for i in range(k):
        I = I + [n + i]
        J = J + [anchorsIdx[i]]
        V = V + [WEIGHT] # default anchor weight

    L = sparse.coo_matrix((V, (I, J)), shape=(n + k, n)).tocsr()

    return L

#Purpose: Given a mesh, to perform Laplacian mesh editing by solving the system
#of delta coordinates and anchors in the least squared sense
#Inputs: mesh (polygon mesh object), anchors (a K x 3 numpy array of anchor
#coordinates), anchorsIdx (a parallel array of the indices of the anchors)
#Returns: Nothing (should update mesh.VPos)
def solveLaplacianMesh(mesh, anchors, anchorsIdx, cotangent=True):
    n = mesh.n_vertices()
    k = anchorsIdx.shape[0]

    operator = (getLaplacianMatrixUmbrella, getLaplacianMatrixCotangent)

    L = operator[1](mesh, anchorsIdx) if cotangent else operator[0](mesh, anchorsIdx)
    delta = np.array(L.dot(mesh.points()))

    # augment delta solution matrix with weighted anchors
    for i in range(k):
        delta[n + i, :] = WEIGHT * anchors[i, :]

    # update mesh vertices with least-squares solution
    for i in range(3):
        #mesh.points()[:, i] = lsqr(L, delta[:, i])[0]
        mesh.points()[:, i] = sparseqr.solve(L, delta[:, i], tolerance = 1e-8)
    
    return mesh



##############################################################
##           High Speed Laplacian Mesh Editing              ##
##############################################################
# using umbrella weights for higher speed
class fast_deform():
    def __init__(self, 
                 f_ijv_pkl = '../predef/dsa_IJV.pkl',
                 f_achr_pkl = '../predef/dsa_achr.pkl',
                 weight = 1.0,
                ):
        self.weight = weight
        with open (f_ijv_pkl, 'rb') as fp:
            dic_IJV = pickle.load(fp)
        I = dic_IJV['I']
        J = dic_IJV['J']
        V = dic_IJV['V']
        self.n = dic_IJV['num_vert']
        with open (f_achr_pkl, 'rb') as fp:
            dic_achr = pickle.load(fp)
        #achr_id = dic_achr['achr_id']
        self.k = dic_achr['achr_num']
        if weight != 1.0:
            num_V = len(V)
            for i in range(num_V-self.k,num_V):
                V[i] = V[i] * self.weight
        self.L = sparse.coo_matrix((V, (I, J)), shape=(self.n + self.k, self.n)).tocsr()
    
    def deform(self, mesh, anchors):
        
        #t_start = time.time()
        delta = np.array(self.L.dot(mesh.points()))
        #t_end = time.time()
        #print("delta computation time is %.5f seconds." % (t_end - t_start))

        #t_start = time.time()
        # augment delta solution matrix with weighted anchors
        for i in range(self.k):
            delta[self.n + i, :] = self.weight * anchors[i, :]
        #t_end = time.time()
        #print("give anchor value computation time is %.5f seconds." % (t_end - t_start))

        #t_start = time.time()
        # update mesh vertices with least-squares solution
        for i in range(3):
            mesh.points()[:, i] = sparseqr.solve(self.L, delta[:, i], tolerance = 1e-8)
            #mesh.points()[:, i] = lsqr(self.L, delta[:, i])[0]
        #t_end = time.time()
        #print("sparse lsqr time is %.5f seconds." % (t_end - t_start))
        
        return mesh

##############################################################
##        High Speed Laplacian Mesh Editing for DSA         ##
##############################################################
class fast_deform_dsa():
    def __init__(self, 
                 f_ijv_pkl = '../predef/dsa_IJV.pkl',
                 f_achr_pkl = '../predef/dsa_achr.pkl',
                 weight = 1.0,
                ):
        self.weight = weight
        with open (f_ijv_pkl, 'rb') as fp:
            dic_IJV = pickle.load(fp)
        self.I = dic_IJV['I']
        self.J = dic_IJV['J']
        self.V = dic_IJV['V']
        self.n = dic_IJV['num_vert']
        with open (f_achr_pkl, 'rb') as fp:
            dic_achr = pickle.load(fp)
        #achr_id = dic_achr['achr_id']
        self.k = dic_achr['achr_num']
        self.num_V = len(self.V)
        if self.weight != 1.0:
            for i in range(self.num_V-self.k, self.num_V):
                self.V[i] = self.V[i] * self.weight
    
    # for inactive index, zero means inactive, non-zeros means active
    def deform(self, verts, achr_verts, active_index = []):
        if active_index != []:
            for i in range(len(active_index)):
                if active_index[i] == 0:
                    self.V[self.num_V-self.k+i] = 0
                    
        self.L = sparse.coo_matrix((self.V, (self.I, self.J)), 
                                   shape=(self.n + self.k, self.n)).tocsr()
        
        delta = np.array(self.L.dot(verts))
        
        # augment delta solution matrix with weighted anchors
        for i in range(self.k):
            delta[self.n + i, :] = self.weight * achr_verts[i, :]
        
        # update mesh vertices with least-squares solution
        deformed_verts = np.zeros(verts.shape)
        for i in range(3):
            deformed_verts[:, i] = sparseqr.solve(self.L, 
                                                  delta[:, i], 
                                                  tolerance = 1e-8
                                                 )
        
        return deformed_verts

##############################################################
##    High Speed Laplacian Mesh Editing for Joint Adapt     ##
##############################################################
class fast_deform_dja():
    def __init__(self, 
                 f_ijv_pkl = '../predef/dja_IJV.pkl',
                 f_achr_pkl = '../predef/dja_achr.pkl',
                 weight = 1.0,
                ):
        self.weight = weight
        with open (f_ijv_pkl, 'rb') as fp:
            dic_IJV = pickle.load(fp)
        self.I = dic_IJV['I']
        self.J = dic_IJV['J']
        self.V = dic_IJV['V']
        self.n = dic_IJV['num_vert']
        with open (f_achr_pkl, 'rb') as fp:
            dic_achr = pickle.load(fp)
        #achr_id = dic_achr['achr_id']
        self.k = dic_achr['achr_num']
        self.num_V = len(self.V)
        if self.weight != 1.0:
            for i in range(self.num_V-self.k, self.num_V):
                self.V[i] = self.V[i] * self.weight
    
    # for inactive index, zero means inactive, non-zeros means active
    def deform(self, verts, achr_verts):
     
        self.L = sparse.coo_matrix((self.V, (self.I, self.J)), 
                                   shape=(self.n + self.k, self.n)).tocsr()
        
        delta = np.array(self.L.dot(verts))
        
        # augment delta solution matrix with weighted anchors
        for i in range(self.k):
            delta[self.n + i, :] = self.weight * achr_verts[i, :]
        
        # update mesh vertices with least-squares solution
        deformed_verts = np.zeros(verts.shape)
        for i in range(3):
            deformed_verts[:, i] = sparseqr.solve(self.L, 
                                                  delta[:, i], 
                                                  tolerance = 1e-8
                                                 )
        
        return deformed_verts
