import open3d as o3d
import numpy as np
# import transformations as tr
from scipy.optimize import least_squares as solver
#from scipy.optimize import leastsq as solver
from transforms3d.euler import euler2mat
import copy
import time

rotation_axis = np.array([0, 0, 1], dtype=float)

def RT(x):
    rot_angle = rotation_axis * x[3]
    M = np.eye(4)
    M[:3, :3] = euler2mat(*np.deg2rad(rot_angle))
    M[0,3]=x[0]
    M[1,3]=x[1]
    M[2,3]=x[2]
    return M

def point_plane_residual(x, source, max_correspond_dist):
    print("x0",x)
    transform = RT(x)
    src=copy.deepcopy(source)
    src.transform(transform)
    points=np.array(src.points)
    nearest_points= []
    nearest_normals= []
    zp= np.array([0, 0, 0], dtype=float)
    for sp in points:
        k, idx, _ = target_kdtree.search_hybrid_vector_3d(sp, max_correspond_dist, 1)
        if k == 1:
            nearest_points.append(target_points[idx[0], :])
            nearest_normals.append(target_normals[idx[0], :])
        else:
            nearest_points.append(sp)
            nearest_normals.append(zp)
    nearest_points=np.array(nearest_points)
    nearest_normals=np.array(nearest_normals)
#    print("nearest_points",nearest_points.shape)
    resvec = points - nearest_points
    residue = np.sum( resvec*nearest_normals, axis=1)  #inner 
#    print("res",len(residue),np.sum(residue!=0))
    return residue

def cicp(pcd_source: o3d.geometry.PointCloud, pcd_target: o3d.geometry.PointCloud, base_transform: np.ndarray,
         max_correspond_dist_coarse, max_correspond_dist_fine):
    global target_kdtree,target_points,target_normals
    if pcd_target.has_normals():
        print("Info source points",len(pcd_source.points))
        print("Info target points",len(pcd_target.points))
    else:
        print("Warning!! No normals")
        o3d.estimate_normals(pcd_target, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    target_kdtree = o3d.geometry.KDTreeFlann(pcd_target)
    target_points=np.array(pcd_target.points)
    target_normals=np.array(pcd_target.normals)
    source=copy.deepcopy(pcd_source)
    source.transform(base_transform)
    x0 = np.array([0, 0, 0, 0], dtype=float)
    result = solver(
        point_plane_residual,
        x0,
        jac='2-point',
        method='trf',
        loss='soft_l1',
        args=(source,max_correspond_dist_coarse)
    )

    x0 = result.x
    result = solver(
        point_plane_residual,
        x0,
        jac='2-point',
        method='trf',
        loss='soft_l1',
        args=(source,max_correspond_dist_fine)
    )

    transform = RT(result.x)
    transform = np.dot(transform, base_transform)

    return transform, result
