from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

@dataclass
class Point(object):

    location: np.ndarray
    value: np.ndarray
    index: np.ndarray
    
@dataclass
class Peak(Point):

    penalty: float
    sign: int

@dataclass
class InteriorPoint(Point):

    gradient: np.ndarray
    hessian: np.ndarray
    tangent_basis: np.ndarray
    n_obs: int # how many fields are observed here -- SHOULD ALWAYS BE 1 NOW
    n_ambient: int # how many spatial dimensions
    n_tangent: int # how many tangent dimensions

@dataclass
class BoundaryPoint(InteriorPoint):

    normal_basis: np.ndarray
    normal_constraint: np.ndarray
    n_normal: int # how many normal dimensions


@dataclass
class InteriorPeak(InteriorPoint, Peak):
    pass

@dataclass
class BoundaryPeak(BoundaryPoint, Peak):
    pass

def get_gradient(pt):
    tangent = get_tangent_gradient(pt)
    normal = get_normal_gradient(pt)
    if pt.n_obs > 1: # pt is a stack of data of 2 or more fields
        return np.concatenate([tangent, normal], axis=1)
    else:
        return np.hstack([tangent, normal])

def get_tangent_gradient(pt):
    if hasattr(pt, 'gradient'):
        G = pt.gradient
        if pt.n_obs > 1: # pt is a stack of data of 2 or more fields
            if hasattr(pt, 'tangent_basis') and pt.tangent_basis is not None: 
                G = G @ pt.tangent_basis.T
        elif hasattr(pt, 'tangent_basis') and pt.tangent_basis is not None:
            G = pt.tangent_basis @ G
    else:
        if pt.n_obs > 1:
            G = np.zeros(shape=(pt.n_obs, 0))
        else:
            G = np.zeros(shape=(0,))
    return G

def get_normal_gradient(pt):
    if hasattr(pt, 'gradient') and hasattr(pt, 'normal_basis'):
        G = pt.gradient
        if pt.n_obs > 1: # pt is a stack of data of 2 or more fields
            if pt.normal_basis is not None: 
                G = G @ normal_basis.T
        elif pt.normal_basis is not None:
            G = pt.normal_basis @ G
    else:
        if hasattr(pt, 'gradient'):
            G = pt.gradient
            if pt.n_obs > 1:
                G = np.zeros(shape=(pt.n_obs, 0))
            else:
                G = np.zeros(shape=(0,))
        else:
            G = np.zeros(shape=(0,))
    return G

def get_hessian(pt):
    if hasattr(pt, 'hessian'):
        H = pt.hessian
        if pt.n_obs > 1: # pt is a stack of data of 2 or more fields
            if hasattr(pt, 'tangent_basis') and pt.tangent_basis is not None: 
                H = np.einsum('ijk,aj,bk->iab',
                              H,
                              pt.tangent_basis,
                              pt.tangent_basis)
        elif hasattr(pt, 'tangent_basis') and pt.tangent_basis is not None:
            H = pt.tangent_basis @ H @ pt.tangent_basis.T
    else:
        H = np.zeros((0,0))
    return H

def extract_peaks(E,
                  clusters,
                  second_order, # second order info (value, ambient gradient, ambient hessian) at each point in E
                  tangent_bases, # basis for tangent space at each point in E
                  normal_info, # normal basis and normal constraint at each point in E
                  locations,
                  signs,
                  penalty_weights,
                  rng=None):

    points, indices = extract_points(E,
                                     clusters,
                                     second_order, 
                                     tangent_bases, 
                                     normal_info, 
                                     locations,
                                     rng=rng)

    # annotate points with sign / penalty info
    # need logic for interior vs. boundary

    peaks = []
    for (point, label) in zip(points, np.unique(clusters)):

        p = point
        cur_cluster = np.nonzero(clusters == label)[0]
        sign = signs[cur_cluster[0]]
        item_ = [tuple(i) for i in zip(*[e[cur_cluster] for e in E])]
        penalty = np.mean([penalty_weights[i] for i in item_])
        peaks.append(annotate_point(point,
                                    sign,
                                    penalty))
    return peaks, indices

def extract_points(E,
                   clusters,
                   second_order, # second order info (value, ambient gradient, ambient hessian) at each point in E
                   tangent_bases, # basis for tangent space at each point in E
                   normal_info, # normal basis and normal constraint at each point in E
                   locations,
                   rng=None):

    if rng is None:
        rng = np.random.default_rng()
    elif type(rng) == int:
        rng = np.random.default_rng(rng)

    points = []
    indices = []

    values = [p[0] for p in second_order]
    gradients = [p[1] for p in second_order]
    hessians = [p[2] for p in second_order]

    dim_error_msg = ('conflicting dimension information in {} for second order' +
                    'approximations at E after clustering, requires inspection' +
                     'to pick solution set E of consistent dimension')

    for lab in np.unique(clusters):
        cur_cluster = np.nonzero(clusters == lab)[0]
                         
        if cur_cluster.shape[0] > 1:
            cur_values = [values[c] for c in cur_cluster]
            value = np.mean(cur_values, 0)

            cur_gradients = [gradients[c] for c in cur_cluster]
            if not np.all([g.shape == cur_gradients[0].shape for g in cur_gradients]):
                raise ValueError(dim_error_msg.format('gradients'))
            gradient = np.mean(cur_gradients, 0)

            cur_hessians = [hessians[c] for c in cur_cluster]
            if not np.all([h.shape == cur_hessians[0].shape for h in cur_hessians]):
                raise ValueError(dim_error_msg.format('hessians'))
            hessian = np.mean(cur_hessians, 0)

            cur_tangent_bases = [tangent_bases[l] for l in cur_cluster]
            if not np.all([b.shape == cur_tangent_bases[0].shape for b in cur_tangent_bases]):
                raise ValueError(dim_error_msg.format('tangent bases'))
            tangent_basis = np.mean(cur_tangent_bases, 0)
            normal_bases = [normal_info[l][0] for l in cur_cluster]
            if not np.all([b.shape == normal_bases[0].shape for b in normal_bases]):
                raise ValueError(dim_error_msg.format('normal bases'))
            normal_basis = np.mean(normal_bases, 0)

            normal_constraints = [normal_info[l][1] for l in cur_cluster]
            if not np.all([b.shape == normal_constraints[0].shape for b in normal_constraints]):
                raise ValueError(dim_error_msg.format('normal constraints'))
            normal_basis = np.mean(normal_constraints, 0)
        else:
            gradient = gradients[cur_cluster[0]]
            hessian = hessians[cur_cluster[0]]
            value = values[cur_cluster[0]]
            normal_basis, normal_constraint = normal_info[cur_cluster[0]]
            tangent_basis = tangent_bases[cur_cluster[0]]
            
        n_ambient = hessian.shape[-1] # might be 0, but that should be ok

        n_obs = value.shape[0]
        value = value.reshape((n_obs,))
        gradient = gradient.reshape((n_obs, n_ambient))
        hessian = hessian.reshape((n_obs, n_ambient, n_ambient))

        idx = rng.choice(cur_cluster, 1)[0]
        if normal_basis.shape[0] == 0:
            cur_points = [InteriorPoint(location=locations[idx].reshape(-1),
                                        value=value[i],
                                        gradient=gradient[i],
                                        hessian=hessian[i],
                                        tangent_basis=tangent_basis,
                                        n_obs=n_obs,
                                        n_ambient=n_ambient,
                                        n_tangent=tangent_basis.shape[0]
                                        ) for i in range(n_obs)]
        else:
            cur_points = [BoundaryPoint(location=locations[idx].reshape(-1),
                                        value=value[i],
                                        gradient=gradient[i],
                                        hessian=hessian[i],
                                        tangent_basis=tangent_basis,
                                        n_obs=n_obs,
                                        n_ambient=n_ambient,
                                        n_tangent=tangent_basis.shape[0],
                                        n_normal=normal_basis.shape[0],
                                        normal_basis=normal_basis,
                                        normal_constraint=normal_constraint
                                        ) for i in range(n_obs)]
            
        points.append(cur_points)
        indices.append(tuple([e[idx] for e in E]))

    return points, np.asarray(indices)

def default_clusters(E,
                     kernel,
                     cor_threshold=0.9):
    K = kernel
    locations = np.array([g[E] for g in K.grid]).T

    if len(locations) > 1:
        HClust = AgglomerativeClustering
        hc = HClust(distance_threshold=1-cor_threshold,
                    n_clusters=None,
                    metric='precomputed',
                    linkage='single')
        C = K.C00(locations,
                  locations)
        diagC = np.diag(C)
        C /= np.multiply.outer(np.sqrt(diagC), np.sqrt(diagC))
        hc.fit(1 - C)
        clusters = hc.labels_
    else:
        clusters = np.array([0])

    E_nz = np.array(np.nonzero(E))
    df = pd.DataFrame({'Cluster': clusters,
                       })
    df['Location'] = [tuple(l) for l in locations]
    df['Index'] = [tuple(E_nz[:,i]) for i in range(E_nz.shape[1])]
    return df


def annotate_point(point,
                   sign,
                   penalty):

    vals = asdict(point)

    vals.update({'sign':sign,
                 'penalty':penalty})

    mapping = {Point: Peak,
               InteriorPoint: InteriorPeak,
               BoundaryPoint: BoundaryPeak}
    cls = mapping[type(point)]
    return cls(**vals)
    # if isinstance(point, Point):
    #     return Peak(**vals)
    # elif isinstance(point, InteriorPoint):
    #     return InteriorPeak(**vals)
    # elif isinstance(point, BoundaryPoint):
    #     return BoundaryPeak(**vals)
