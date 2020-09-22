"""
This file consists of the helper functions to MathematicalMorphologyAndDataAnalysisV*.ipynb
"""

#-----------------------------------------------------------------------------------------------#
#------------------------------------------ UNIONFIND ------------------------------------------#
#-----------------------------------------------------------------------------------------------#

import numpy as np

class unionfind:
    
    def __init__(self, size_data):
        self.size_data = size_data
        self.parent = np.arange(size_data, dtype=np.int32)
        self.size = np.ones(size_data, dtype=np.int32)
        self.comp = dict([(x, set([x])) for x in range(size_data)])
        
    
    def find(self, x):
        """ Return the parent of x
        """
        par = x
        list_par = []
        
        while par != self.parent[par]:
            list_par.append(par)
            par = self.parent[par]
            
        for elt in list_par:
            self.parent[elt] = par
            
        return par
    
    def union(self, x, y):
        """ Combine the components of x and y
        """
        
        px, py = self.find(x), self.find(y)
        if px != py:
            
            if self.size[px] > self.size[py]:
                px, py = py, px
                

            self.parent[px] = py
            self.size[py] += self.size[px]
            
    def generate_comps(self):
        """Generate the components
        """
        comp = np.array([self.find(x) for x in range(self.size_data)])
        for l in np.unique(comp):
            yield l, np.where(comp==l)[0]
    

#----------------------------------------------------------------------------------------------#
#------------------------------------ WATERSHED SUPERVISED ------------------------------------#
#----------------------------------------------------------------------------------------------#

import numpy as np
import scipy as sp

def watershed_supervised(graph, seeds):
    """ Return the labelling obtained by MSF-watershed
    """
    
    size_data = graph.shape[0]
    
    u, v, w = sp.sparse.find(graph)
    list_edges = list(zip(u,v,w))
    list_edges.sort(key = lambda x : x[2])
    
    UF = unionfind(size_data)
    labels = np.array(seeds, dtype=np.int32, copy=True)
    
    for e in list_edges:
        ex, ey, ew = e
        px, py = UF.find(ex), UF.find(ey)
        if px != py:
            lx, ly = labels[px], labels[py]
            if (lx != 0) and (ly !=0 ):
                # Both are labelled. Do nothing
                pass
            else:
                max_label = max(lx, ly) # Pick the non-zero label!!
                labels[px], labels[py] = max_label, max_label
                UF.union(ex, ey)
                
                
    ans = np.array([labels[UF.find(x)] for x in range(size_data) ] )
    
#     assert np.all(ans>0)
    
    return ans
    
    
    
#----------------------------------------------------------------------------------------------#
#---------------------------------- PLOTTING THE BOUNDARIES -----------------------------------#
#----------------------------------------------------------------------------------------------#

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN

def plot_boundary(X, y, resolution=100, n_neighbors=1):
    """ Returns the points which should be plotted! 
    """
    
    xmin, xmax, ymin, ymax = np.min(X[:,0]), np.max(X[:,0]), np.min(X[:,1]), np.max(X[:,1])
    
    xs, ys = np.linspace(xmin-0.1, xmax+0.1, num=resolution), np.linspace(ymin-0.1, ymax+0.1, num=resolution)
    xgrid, ygrid = np.meshgrid(xs, ys)
    
    
    clf = KNN(n_neighbors=n_neighbors)
    clf.fit(X, y)
    
    Xpred = np.stack((xgrid.flatten(), ygrid.flatten()), axis=1)
    ypred = clf.predict(Xpred)
    ypred = ypred.reshape((resolution, resolution))
    
    ind1 = np.where(ypred[:-1,:] != ypred[1:,:])
    ind2 = np.where(ypred[:,:-1] != ypred[:,1:])
    
    xret = np.concatenate((xgrid[ind1].flatten(), xgrid[ind2].flatten()))
    yret = np.concatenate((ygrid[ind1].flatten(), ygrid[ind2].flatten()))
    
    return xret, yret
    
    
    
#------------------------------------------------------------------------------------#
#------------------------------- CONSTRUCT 4ADJ GRAPH -------------------------------#
#------------------------------------------------------------------------------------#

import scipy as sp    
    
def construct_4adj_graph(X):
    """Construct the 4-adjacency graph from the nd-array
    """
    s0, s1, s2 = X.shape
    
    size_data = s0*s1
    
    xGrid, yGrid = np.meshgrid(np.arange(s0), np.arange(s1))
    totGrid = (xGrid*s1 + yGrid).transpose()
    
    horiz_edges = np.sqrt(np.sum((X[1:,:,:] - X[:-1,:,:])**2, axis=-1).flatten())
    indx_horiz_edges = totGrid[1:,:].flatten()
    indy_horiz_edges = totGrid[:-1,:].flatten()
    
    vert_edges = np.sqrt(np.sum((X[:,1:,:] - X[:,:-1,:])**2, axis=-1).flatten())
    indx_vert_edges = totGrid[:,1:].flatten()
    indy_vert_edges = totGrid[:,:-1].flatten()
    
    w = np.concatenate((horiz_edges, vert_edges), axis=0) + 1e-6
    u = np.concatenate((indx_horiz_edges, indx_vert_edges), axis=0)
    v = np.concatenate((indy_horiz_edges, indy_vert_edges), axis=0)
    
    return sp.sparse.csr_matrix((w,(u,v)), shape=(size_data, size_data))



#-------------------------------------------------------------------------------------#
#------------------------------- MAKE GRAPH UNDIRECTED -------------------------------#
#-------------------------------------------------------------------------------------#

import scipy as sp            
def make_undirected(G):
    """This function takes the graph and returns the undirected version.
    """
    u,v,w = sp.sparse.find(G)
    
    edges = dict()
    for i in range(u.shape[0]):
        edges[(u[i],v[i])] = w[i]
        edges[(v[i],u[i])] = w[i]
        
   
    sizeNew = len(edges)
    uNew = np.zeros(sizeNew, dtype=np.int32)
    vNew = np.zeros(sizeNew, dtype=np.int32)
    wNew = np.zeros(sizeNew, dtype=np.float64)
    
    i = 0
    for ((u,v),w) in edges.items():
        uNew[i], vNew[i], wNew[i] = u, v, w
        i += 1
        
    assert i == sizeNew, "Something went wrong"
    
    return sp.sparse.csr_matrix((wNew,(uNew,vNew)), shape=G.shape)



#-------------------------------------------------------------------------------------#
#-------------------------------- ENSEMBLE WATERSHEDS --------------------------------#
#-------------------------------------------------------------------------------------#

import scipy as sp

def ensemble_watershed(X, graph, seeds, number_estimators=10, num_features_select=20, percentage_seed_select = 20, gt=None):
    """
    """
    
    size_data = X.shape[0]
    num_features = X.shape[1]
    
    classes = np.unique(seeds[seeds > 0])
    num_classes = classes.shape[0]
    
    labelling = np.zeros((size_data, num_classes), dtype=np.float64)
    
    history = []
    
    for est in range(number_estimators):
        
        # Select a subset of features
        indSelect_features = np.random.choice(np.arange(num_features), size=num_features_select)
        X_tmp = X[:,indSelect_features]
        
        # Select a subset of labelled points as seeds
        seed_tmp = np.zeros(size_data)
        for c in classes:
            indSelect_seeds = np.random.choice(np.where(seeds == c)[0], size=int(0.01*percentage_seed_select*np.sum(seeds==c)))
            seed_tmp[indSelect_seeds] = seeds[indSelect_seeds]
            
        
        # Construct the edge weighted graph
        u,v,w = sp.sparse.find(graph)
        wNew = np.sum((X_tmp[u,:] - X_tmp[v,:])**2, axis=1) + 1e-6
        graph_tmp = sp.sparse.csr_matrix((wNew,(u,v)), shape=graph.shape)
        
        
        # Get the watershed labelling
        labels_tmp = watershed_supervised(graph_tmp, seed_tmp)
        
        # Add the accuracy of the labels to the labelling
        indSelect = np.where(np.logical_and(seeds > 0, seed_tmp==0))
        acc = np.mean(seeds[indSelect] == labels_tmp[indSelect])
        for i in range(num_classes):
            ind_points = np.where(labels_tmp == classes[i])
            labelling[ind_points,i] += acc
            

        try:
            L = np.argmax(labelling, axis=1)
            acc = np.mean((classes[L[seeds==0]]-1) == gt[seeds==0])
            history.append(acc)
            print('\rCurrent accuracy after iter {} : '.format(est), acc, end="")
        except:
            pass
            
    L = np.argmax(labelling, axis=1)          
    return classes[L], history


#--------------------------------------------------------------------------------------#
#-------------------------------------- NN MODEL --------------------------------------#
#--------------------------------------------------------------------------------------#

# from keras.layers import Dense, Input, Activation, Lambda, Dropout
# from keras.models import Model
# from keras.optimizers import Adam, RMSprop
# import keras.backend as K

# def euclidean_distance(vects):
#     """Return the eucliden distance between the inputs
#     """
#     x, y = vects
#     sum_square = K.sum(K.square(x-y), axis=1, keepdims=True)
#     return K.sqrt(K.maximum(sum_square, K.epsilon()))

# def base_model(input_size):
    
#     inp = Input((input_size,))
#     out = Dense(128, activation='relu')(inp)
#     out = Dense(64, activation='relu')(out)
#     out = Dense(32, activation='relu')(out)    
#     M = Model(inp, out)
    
#     return M

# def model(input_size):
    
#     M_rep = base_model(input_size)
    
#     inpa = Input((input_size,))
#     outa = M_rep(inpa)
    
#     inpb = Input((input_size,))
#     outb = M_rep(inpb)
    
#     out = Lambda(euclidean_distance)([outa, outb])
#     out = Dense(1, activation='sigmoid')(out)
    
#     M = Model([inpa, inpb], out )
    
#     opt = Adam(lr=0.01)
#     M.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
#     return M, M_rep
    

#--------------------------------------------------------------------------------------#
#-------------------------------------- NN MODEL V2 --------------------------------------#
#--------------------------------------------------------------------------------------#


# def euclidean_distance(vects):
#     x, y = vects
#     sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
#     return K.sqrt(K.maximum(sum_square, K.epsilon()))


# def eucl_dist_output_shape(shapes):
#     shape1, shape2 = shapes
#     return (shape1[0], 1)


# def contrastive_loss(y_true, y_pred):
#     '''Contrastive loss from Hadsell-et-al.'06
#     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     '''
#     margin = 1
#     sqaure_pred = K.square(y_pred)
#     margin_square = K.square(K.maximum(margin - y_pred, 0))
#     return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

# def compute_accuracy(y_true, y_pred):
#     '''Compute classification accuracy with a fixed threshold on distances.
#     '''
#     pred = y_pred.ravel() < 0.5
#     return np.mean(pred == y_true)


# def accuracy(y_true, y_pred):
#     '''Compute classification accuracy with a fixed threshold on distances.
#     '''
#     return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

# def base_modelV2(input_size):
#     """ Base network
#     """
#     inp = Input((input_size,))
#     x = Dense(1024, activation='relu')(inp)
#     x = Dense(512, activation='relu')(x)
#     x = Dense(256, activation='relu')(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dense(64, activation='relu')(x)
#     x = Dense(32, activation='relu')(x)
#     return Model(inp, x)


# def modelV2(input_size):
    
#     M_rep = base_modelV2(input_size)
    
#     inpa = Input((input_size,))
#     outa = M_rep(inpa)
    
#     inpb = Input((input_size,))
#     outb = M_rep(inpb)
    
#     out = Lambda(euclidean_distance)([outa, outb])
# #     out = Dense(1, activation='sigmoid')(out)
    
#     M = Model([inpa, inpb], out )
    
#     opt = Adam(lr=0.0005)
#     M.compile(optimizer=opt, loss=contrastive_loss, metrics=[accuracy])
    
#     return M, M_rep



#-----------------------------------------------------------------------------------------#
#-------------------------------------- NN MODEL V3 --------------------------------------#
#-----------------------------------------------------------------------------------------#


# def euclidean_distance(vects):
#     x, y = vects
#     sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
#     return K.sqrt(K.maximum(sum_square, K.epsilon()))


# def eucl_dist_output_shape(shapes):
#     shape1, shape2 = shapes
#     return (shape1[0], 1)


# def contrastive_loss(y_true, y_pred):
#     '''Contrastive loss from Hadsell-et-al.'06
#     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     '''
#     margin = 1
#     sqaure_pred = K.square(y_pred)
#     margin_square = K.square(K.maximum(margin - y_pred, 0))
#     return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

# def compute_accuracy(y_true, y_pred):
#     '''Compute classification accuracy with a fixed threshold on distances.
#     '''
#     pred = y_pred.ravel() < 0.5
#     return np.mean(pred == y_true)


# def accuracy(y_true, y_pred):
#     '''Compute classification accuracy with a fixed threshold on distances.
#     '''
#     return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

# def base_modelV3(input_size):
#     """ Base network
#     """
#     inp = Input((input_size,))
#     x = Dense(128, activation='relu')(inp)
#     x = Dense(128, activation='relu')(x)
#     x = Dense(128, activation='relu')(x)
#     return Model(inp, x)


# def modelV3(input_size):
    
#     M_rep = base_modelV3(input_size)
    
#     inpa = Input((input_size,))
#     outa = M_rep(inpa)
    
#     inpb = Input((input_size,))
#     outb = M_rep(inpb)
    
#     out = Lambda(euclidean_distance)([outa, outb])
# #     out = Dense(1, activation='sigmoid')(out)
    
#     M = Model([inpa, inpb], out )
    
#     opt = Adam(lr=0.0005)
#     M.compile(optimizer=opt, loss=contrastive_loss, metrics=[accuracy])
    
#     return M, M_rep

#----------------=-----------------------------------------------------------------------#
#------------------------------------- SALIENCY MAP -------------------------------------#
#----------------------------------------------------------------------------------------#

import scipy as sp

class saliencymap:
    
    def __init__(self, graph):
        
        self.size_data =  graph.shape[0]
        self.parent = np.arange(self.size_data, dtype=np.int32)
        self.compsize = np.ones(self.size_data, dtype=np.int32)
        self.graph = graph
        
        # Also define the boundary
        indices, indptr = graph.indices, graph.indptr
        self.boundary = dict([(x,set()) for x in range(self.size_data)])
        for x in range(self.size_data):
            for i in range(indptr[x], indptr[x+1]):
                ex, ey = min(x, indices[i]), max(x, indices[i])
                self.boundary[ex].add((ex,ey))
                self.boundary[ey].add((ex,ey))
        
        saliency_map = None
        
    def find(self, x):
        """
        """
        par = x
        list_par = []
        
        while par != self.parent[par]:
            list_par.append(par)
            par = self.parent[par]
            
        for elt in list_par:
            self.parent[elt] = par
            
        return par
    
    def union(self, x, y):
        """
        """
        px, py = self.find(x), self.find(y)
        
        if px != py:
            
            if self.size[px] > self.size[py]:
                px, py = py, px
                
            self.parent[px] = py
            self.size[py] += self.size[px]
            
            self.boundary[py]=self.boundary[py].symmetric_difference_update(self.boundary[px])
            
            del self.boundary[px]
            
            
        
    def construct_saliency_map(self):
        """
        """
        
        self.saliency_map = sp.sparse.dok_matrix(self.graph)
        
        u,v,w = sp.sparse.find(self.graph)
        list_edges = list(zip(u,v,w))
        list_edges.sort(key = lambda x : x[2])
        
        for e in list_edges:
            ex, ey, ew = e
            px, py = self.find(ex), self.find(ey)
            if px != py:
                for e in self.boundary[px].intersection(self.boundary[py]):
                    e0, e1 = e
                    self.saliency_map[e0,e1] = ew
                    self.saliency_map[e1,e0] = ew
                    
        self.saliency_map = sp.sparse.csr_matrix(self.saliency_map)
            


def construct_saliency_map(graph):
    """Returns the graph with edge weights replaced by pass value
    """
    
    SP = saliencymap(graph)
    SP.construct_saliency_map()
    return SP.saliency_map
    
#----------------=-----------------------------------------------------------------------#
#-------------------------------- VISUALIZE SALIENCY MAP --------------------------------#
#----------------------------------------------------------------------------------------#    
    
def visualize_saliency_map(saliency_map, s0, s1):
    """
    """
    
    xgrid, ygrid = np.meshgrid(np.arange(s0), np.arange(s1))
    grid = (xgrid*s1 + ygrid).transpose()
    
    
    img = np.zeros((2*s0+1, 2*s1+1), dtype=np.float64)
    
    indices, indptr, data = saliency_map.indices, saliency_map.indptr, saliency_map.data
    for x in range(s0*s1):
        for i in range(indptr[x],indptr[x+1]):
            e0, e1, e2 = x, indices[i], data[i]
            e0x, e0y = e0//s1, e0%s1
            e1x, e1y = e1//s1, e1%s1
            sx, sy = e0x + e1x + 1, e0y + e1y + 1
            img[sx, sy] = e2
                       
                       
    for x in range(2,2*s0-1,2):
        for y in range(2,2*s1-1,2):
            img[x,y] = max(img[x-1,y], img[x+1,y], img[x,y-1], img[x,y+1])
            
    
    return img

#----------------=-----------------------------------------------------------------------#
#------------------------------- MAXIMUM MARGIN CLASSIFIER -------------------------------#
#----------------------------------------------------------------------------------------#   

import scipy as sp
from sklearn.neighbors import KNeighborsClassifier as KNN

def maximum_margin_classifier(X, graph, seeds):
    """
    """
    
    size_data = graph.shape[0]
    list_classes = np.unique(seeds[seeds>0])
    num_classes = list_classes.shape[0]

    u,v,w = sp.sparse.find(graph)
    list_edges = list(zip(u,v,w))
    list_edges.sort(key = lambda x:x[2])
    
    margin = np.zeros(num_classes+1, dtype=np.float64)
    UF = unionfind(size_data)
    
    labels = np.array(seeds, dtype=np.int32, copy=True)
    
    max_margin = 0
    opt_thresh = None
    for e in list_edges:
        ex,ey,ew = e
        px, py = UF.find(ex), UF.find(ey)
        if px!=py:
            if (labels[px] != 0) and (labels[py] != 0):
                # Both components are labelled and
                # the current margin for both classes is less than
                # this value!!!
                margin[labels[py]] = max(margin[labels[py]],ew)
                margin[labels[px]] = max(margin[labels[px]],ew)
            elif (labels[px] == 0) and (labels[py] == 0):
                UF.union(ex,ey)
            else:
                # Assume that label[px] = 0
                if labels[px] != 0:
                    px, py = py, px
                    
                UF.union(ex,ey)
                margin[labels[py]] = max(margin[labels[py]],ew)
                

        if max_margin < np.min(margin[1:]):
            max_margin = np.min(margin[1:])+1e-8
            opt_thresh = ew + 1e-8
            
    Gtmp = sp.sparse.csr_matrix(graph, copy=True)
    Gtmp.data[Gtmp.data > opt_thresh] = 0
    Gtmp.eliminate_zeros()
    n, label_comp = sp.sparse.csgraph.connected_components(Gtmp)
        
    Xcomp = np.zeros((n, X.shape[1]), dtype=np.float64)
    for l in range(n):
        Xcomp[l,:] = np.median(X[np.where(label_comp==l)], axis=0)
        
    clf = KNN(n_neighbors=1)
    clf.fit(X[seeds>0], seeds[seeds>0])
    labels_tmp = clf.predict(Xcomp)
    
    
    return opt_thresh, max_margin, labels_tmp[label_comp]
            
        
#----------------=------------------------------------------------------------------------#
#----------------------------------- ACCURACY MEASURES -----------------------------------#
#-----------------------------------------------------------------------------------------#       

from sklearn.metrics import cohen_kappa_score
import numpy as np

def get_acc_measure(ytrue, ypred):
    
    overall_acc = np.mean(ytrue==ypred)
    
    avg_acc_list = []
    for lab in np.unique(ytrue):
        ind = np.where(ytrue==lab)
        avg_acc_list.append(np.mean(ypred[ind]==lab))
        
    average_acc = np.mean(avg_acc_list)
    
    kappa = cohen_kappa_score(ytrue, ypred)
    
    return overall_acc, average_acc, kappa
    
        



#------------------------------------------------------------------------------------------#
#------------------------------------- POWERWATERSHED -------------------------------------#
#------------------------------------------------------------------------------------------#


import numpy as np
from scipy.sparse import find, csr_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import cg, minres
from scipy.linalg import norm
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import connected_components

import pdb

def generate_edges(graph, bucketing='epsilon', eps=1e-2, k=3):
    """Generate the set of edges
    """
    u, v, w = find(graph)

    if bucketing == 'epsilon':
        w_prime = np.array(w/eps, dtype=np.int32)

    elif bucketing == 'kmeans':
        clf = KMeans(n_clusters=k)
        clf.fit(w.reshape((-1, 1)))
        centers, labels = clf.cluster_centers_, clf.labels_
        w_prime = centers[labels]

    list_edges = list(zip(u, v, w, w_prime))
    list_edges.sort(key=lambda x: -x[2])

    cur_weight = list_edges[0][3]
    edges_bucket = []
    for e in list_edges:
        if np.abs(e[3] - cur_weight) < 1e-7:
            edges_bucket.append(e)
        else:
            yield edges_bucket
            edges_bucket = []
            cur_weight = e[3]

    yield edges_bucket  # to return the last bucket!


def _randomwalk(list_edges, labels, comp, verbose=False):

    size = labels.shape[0]

    u, v, w, w_prime = (zip(*list_edges))
    G = csr_matrix((w, (u, v)), shape=(size, size))
    G = G[comp][:, comp]
    G = make_undirected(G)

    L = csr_matrix(csgraph.laplacian(G))
    lab_comp = labels[comp]

    # Compute the indices of labelled and unlabelled points
    indices = np.arange(lab_comp.size, dtype=np.int32)
    unlabeled_indices = np.array(indices[np.where(lab_comp < 0)], dtype=np.int32)
    seeds_indices = np.array(indices[np.where(lab_comp >= 0)], dtype=np.int32)

    # Split the laplacian
    B = L[unlabeled_indices][:, seeds_indices]
    lap_sparse = L[unlabeled_indices][:, unlabeled_indices]

    # Calculate the R.H.S
    rhs1 = np.array(lab_comp[seeds_indices], dtype=np.float64)
    rhs1 = rhs1.reshape((-1, 1))
    rhs = -1*B.dot(rhs1)

    x0, info = cg(lap_sparse, rhs, tol=1e-3, maxiter=2000)

    if verbose:
        if norm(rhs) > 1e-2:
            print("\rPython solver relative error is {:02.4f}".format(
                norm(lap_sparse.dot(x0.reshape(-1, 1))-rhs)/norm(rhs)), end="")
        else:
            print("\rPython solver relative error is {:02.4f}".format(
                norm(lap_sparse.dot(x0.reshape(-1, 1))-rhs)), end="")

    labels[comp[unlabeled_indices]] = np.ravel(x0)


def powerWatershed(graph, seeds, bucketing='kmeans', eps=1e-2, k=3, beta=5., eps_weight=1e-6):
    """Returns the power watershed labelling given the seeds

    INPUT
    -----
    graph: csr matrix
        similarity adjacency matrix
    seeds: 1d array
        Array indicating the seeds. 0 indicates unlabelled points
    bucketing: one of ['epsilon', 'kmeans']
        bucketing procedure to use
    """

    size = graph.shape[0]

    graph_tmp = csr_matrix(graph, copy=True)
    graph_tmp.data = np.exp(-1*beta*graph_tmp.data/graph_tmp.data.std()) + eps_weight
    
    labels = np.array(seeds, dtype=np.float64) - 1

    uf = unionfind(size)

    edge_generator = generate_edges(graph, bucketing, eps, k)
    edges_till_now = []
    for edges in edge_generator:

        # Add all the edges to the graph
        for e in edges:
            uf.union(e[0], e[1])

        edges_till_now += edges

        for (_, comp) in uf.generate_comps():
            label_unique = set(np.unique(labels[comp]))

            # continue if all points in the component are labelled
            if -1 not in label_unique:
                continue

            # othwerwise solve rw
            if len(label_unique) <= 2:
                l = max(label_unique)
                labels[comp] = l

            if len(label_unique) > 2:
                _randomwalk(edges_till_now, labels, comp)

    return labels


def powerWatershed_multipleLabels(graph, seeds, bucketing='kmeans', eps=1e-2, k=3, beta=5., eps_weight=1e-6):
    """Returns the power watershed labelling given the seeds and multiple Labels

    INPUT
    -----
    graph: csr matrix
        similarity adjacency matrix
    seeds: 1d array
        Array indicating the seeds. 0 indicates unlabelled points
    bucketing: one of ['epsilon', 'kmeans']
        bucketing procedure to use
    """

    ans = []
    for i in np.sort(np.unique(seeds)):
        if i > 0:
            seed_tmp = np.array((seeds == i)*1, dtype=np.int32) + 1
            seed_tmp[np.where(seeds == 0)] = 0
            ans.append(powerWatershed(graph, seed_tmp, bucketing, eps, k, beta, eps_weight))
    return np.argmax(ans, 0) + 1




#---------------------------------------------------------------------------------------------#
#---------------------------------------- RANDOM WALK ----------------------------------------#
#---------------------------------------------------------------------------------------------#

from scipy.sparse.linalg import cg
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix

def _buildAB(lap_sparse, labels):
    """
    Build the matrix A and rhs B of the linear system to solve.
    A and B are two block of the laplacian of the image graph.
    """
    labels = labels[labels >= 0]
    indices = np.arange(labels.size)
    unlabeled_indices = indices[labels == 0]
    seeds_indices = indices[labels > 0]
    
    # The following two lines take most of the time in this function
    B = lap_sparse[unlabeled_indices][:,seeds_indices]
    lap_sparse = lap_sparse[unlabeled_indices][:,unlabeled_indices]
    
    nlabels = labels.max()
    rhs = []
    for lab in range(1, nlabels + 1):
        mask = (labels[seeds_indices] == lab)
        fs = csr_matrix(mask)
        fs = fs.transpose()
        rhs.append(B * fs)
    return lap_sparse, rhs


def _solve_cg(lap_sparse, B, tol, return_full_prob=False):
    """
    solves lap_sparse X_i = B_i for each phase i, using the conjugate
    gradient method. For each pixel, the label i corresponding to the
    maximal X_i is returned.
    """
    lap_sparse = lap_sparse.tocsc()
    X = []
    for i in range(len(B)):
        x0 = cg(lap_sparse, -B[i].toarray(), tol=tol)[0]
        X.append(x0)
    if not return_full_prob:
        X = np.array(X)
        X = np.argmax(X, axis=0)
    return X

def RW(graph, seeds, beta=5., eps=1e-6):
    """
    """
    
    graph_tmp = csr_matrix(graph, copy=True)
    graph_tmp.data = np.exp(-1*beta*graph_tmp.data/graph_tmp.data.std()) + eps
    
    lap = csr_matrix(laplacian(graph_tmp, normed=False))
    A, B = _buildAB(lap, np.array(seeds,dtype=np.int32))
    labels_tmp = _solve_cg(A, B, tol=1e-6, return_full_prob=False)
    
    labels = np.array(seeds)
    labels[seeds == 0] = labels_tmp+1
    return labels


#--------------------------------------------------------------------------------------------#
#-------------------------------------- PRIORITY QUEUE --------------------------------------#
#--------------------------------------------------------------------------------------------#

import heapq
import itertools

class priorityQueue:
    
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = "REMOVED"
        self.counter = itertools.count()
        
    def add_element(self, x, priority=(0,0)):
        if x in self.entry_finder:
            self.remove_element(x)
        count = next(self.counter)
        entry = [priority, count, x]
        self.entry_finder[x] = entry
        heapq.heappush(self.pq, entry)
        
    def pop_element(self):
        while self.pq:
            priority, count, x = heapq.heappop(self.pq)
            if x != 'REMOVED':
                del self.entry_finder[x]
                return x, priority
        raise KeyError('pop from empty queue')
        
    def remove_element(self, x):
        if x in self.entry_finder:
            entry = self.entry_finder[x]
            entry[-1] = self.REMOVED

#---------------------------------------------------------------------------------------------#
#--------------------------------- Image Foresting Transform ---------------------------------#
#---------------------------------------------------------------------------------------------#

import numpy as np

def _compare(t_sum, t_max, d_sum, d_max, alg='SP-SUM'):
    """
    """
    if alg == 'SP-SUM':
        return t_sum < d_sum
    elif alg == 'SP-MAX':
        return t_max < d_max
    elif alg == 'SP-COMB':
        return (t_max, t_sum) < (d_max, d_sum)
    else:
        raise Exception("alg not recognized. Should be one of 'SP-SUM', 'SP-MAX' or 'SP-COMB'")


def IFT(graph, seeds, alg='SP-SUM'):
    """Returns the labelling using shortest paths
    """
    
    indices, indptr, data = graph.indices, graph.indptr, graph.data
    
    # Initialize the required variables
    size_data = graph.shape[0]
    
    cost_sum = np.inf*np.ones(size_data, dtype=np.float64)
    cost_sum[seeds > 0] = 0 # cost is 0 for labelled points
    
    cost_max = np.inf*np.ones(size_data, dtype=np.float64)
    cost_max[seeds > 0] = 0 # cost is 0 for labelled points
    
    labels = np.array(seeds, dtype=np.int32)
    
    pq = priorityQueue()
    
    # Add elements to priority queue
    for i in range(size_data):
        if seeds[i] > 0:
            pq.add_element(i, priority=(0,0))
            
    while len(pq.pq) > 0:
        try:
            x, priority = pq.pop_element()
        except KeyError:
            print("priority queue is empty")
            break
            
        for i in range(indptr[x],indptr[x+1]):
            y = indices[i]
            tmp_cost_sum = cost_sum[x] + data[i]
            tmp_cost_max = max(cost_max[x], data[i])
            
            if _compare(tmp_cost_sum, tmp_cost_max, cost_sum[y], cost_sum[y], alg):
                
                if cost_sum[y] < np.inf:
                    pq.remove_element(y)
                
                cost_sum[y] = tmp_cost_sum
                cost_max[y] = tmp_cost_max
                labels[y] = labels[x]
                
                pq.add_element(y, priority=(tmp_cost_max, tmp_cost_sum))
                

    assert np.all(labels > 0), "Not all points are labelled. Is the graph connected???"
    
    return labels


#---------------------------------------------------------------------------------------------#
#--------------------------------------- LOAD DATASETS ---------------------------------------#
#---------------------------------------------------------------------------------------------#

import numpy as np
from scipy.sparse import csr_matrix
import scipy as sp
from scipy.io import loadmat, savemat
from scipy.sparse.csgraph import connected_components

from sklearn.neighbors import kneighbors_graph

def get_data(name):
    """Returns the edge weighted graph according to the dataset.
    """
    
    if name == "SSL1":
        data = loadmat("./Data/SSL_datasets/SSL,set=1,data.mat")
        X, y = data['X'], data['y']
        y[y == -1] = 0
        y += 1 # Adding one to be consistent with the code
        graph = kneighbors_graph(X, n_neighbors=30, mode='distance')
        graph = make_undirected(graph)
        n, _ = connected_components(graph)
        assert n==1, "Number of connected components {} > 1".format(n)
        return X, y, graph
    elif name == "SSL2":
        data = loadmat("./Data/SSL_datasets/SSL,set=2,data.mat")
        X, y = data['X'], data['y']
        y[y == -1] = 0
        y += 1 # Adding one to be consistent with the code
        graph = kneighbors_graph(X, n_neighbors=30, mode='distance')
        graph = make_undirected(graph)
        n, _ = connected_components(graph)
        assert n==1, "Number of connected components {} > 1".format(n)
        return X, y, graph
    elif name == "SSL3":
        data = loadmat("./Data/SSL_datasets/SSL,set=3,data.mat")
        X, y = data['X'], data['y']
        y[y == -1] = 0
        y += 1 # Adding one to be consistent with the code
        graph = kneighbors_graph(X, n_neighbors=80, mode='distance')
        graph = make_undirected(graph)
        n, _ = connected_components(graph)
        assert n==1, "Number of connected components {} > 1".format(n)
        return X, y, graph
    elif name == "SSL4":
        data = loadmat("./Data/SSL_datasets/SSL,set=4,data.mat")
        X, y = data['X'], data['y']
        y[y == -1] = 0
        y += 1 # Adding one to be consistent with the code
        graph = kneighbors_graph(X, n_neighbors=30, mode='distance')
        graph = make_undirected(graph)
        n, _ = connected_components(graph)
        assert n==1, "Number of connected components {} > 1".format(n)
        return X, y, graph
    elif name == "SSL5":
        data = loadmat("./Data/SSL_datasets/SSL,set=5,data.mat")
        X, y = data['X'], data['y']
        y[y == -1] = 0
        y += 1 # Adding one to be consistent with the code
        graph = kneighbors_graph(X, n_neighbors=30, mode='distance')
        graph = make_undirected(graph)
        n, _ = connected_components(graph)
        assert n==1, "Number of connected components {} > 1".format(n)
        return X, y, graph    
    elif name == "SSL6":
        data = loadmat("./Data/SSL_datasets/SSL,set=6,data.mat")
        X, y = data['X'], data['y']
        y[y == -1] = 0
        y += 1 # Adding one to be consistent with the code
        graph = kneighbors_graph(X, n_neighbors=40, mode='distance')
        graph = make_undirected(graph)
        n, _ = connected_components(graph)
        assert n==1, "Number of connected components {} > 1".format(n)
        return X, y, graph 
    elif name == "SSL7":
        data = loadmat("./Data/SSL_datasets/SSL,set=7,data.mat")
        X, y = data['X'], data['y']
        y[y == -1] = 0
        y += 1 # Adding one to be consistent with the code
        graph = kneighbors_graph(X, n_neighbors=30, mode='distance')
        graph = make_undirected(graph)
        n, _ = connected_components(graph)
        assert n==1, "Number of connected components {} > 1".format(n)
        return X, y, graph
    else:
        raise Exception("Invalid dataset {}".format(name))