import open3d as o3d
import numpy as np
from scipy.optimize import least_squares as solver
from scipy.spatial.transform import Rotation as Rot
try:
  from rovi_utils import tflib
except:
  import tflib
import yaml
import copy

def getRT(base,ref):
  try:
    ts=tfBuffer.lookup_transform(base,ref,rospy.Time())
    rospy.loginfo("cropper::getRT::TF lookup success "+base+"->"+ref)
    RT=tflib.toRT(ts.transform)
  except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    RT=None
  return RT

def RT(x):
    M = np.eye(4)
    M[:3, :3] = Rot.from_euler('xyz',[0,0,x[3]],degrees=True).as_matrix()
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
######codes below will be replaced to a pybind module######
    residue= []
    zp= np.array([0, 0, 0], dtype=float)
    for sp in points:
        k, idx, _ = target_kdtree.search_hybrid_vector_3d(sp, max_correspond_dist, 1)
        if k == 1:
            i=idx[0]
            residue.append(target_normals[i].dot(sp-target_points[i]))
        else:
            residue.append(0)
######residue returned########################
    return np.array(residue)

def cicp(pcd_source: o3d.geometry.PointCloud, pcd_target: o3d.geometry.PointCloud,max_correspond_dist_coarse, max_correspond_dist_fine,precision):
    global target_kdtree,target_points,target_normals
    if pcd_target.has_normals():
        print("Info source points",len(pcd_source.points))
        print("Info target points",len(pcd_target.points))
    else:
        print("Warning!! No normals")
        o3d.estimate_normals(pcd_target, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

######codes below will be replaced to a pybind module######
    target_points=np.array(pcd_target.points)                          #target_points and target_normals
    target_normals=np.array(pcd_target.normals)                    #should be preset to the module
    target_kdtree = o3d.geometry.KDTreeFlann(pcd_target)    #kdtree will be kept in the module
######nothing returned########################
    source=copy.deepcopy(pcd_source)
    x0 = np.array([0, 0, 0, 0], dtype=float)
    result = solver(
        point_plane_residual,
        x0,
        jac='2-point',
        method='trf',
        ftol=precision,
        loss='soft_l1',
        args=(source,max_correspond_dist_coarse)
    )

    x0 = result.x
    print("Fine ICP")
    result = solver(
        point_plane_residual,
        x0,
        jac='2-point',
        method='trf',
        ftol=precision,
        loss='soft_l1',
        args=(source,max_correspond_dist_fine)
    )

    transform = RT(result.x)
    return transform, result

def learn(datArray,param):
  global pcd_source,kd_param
  print("CGA solver learn")
  kd_param=o3d.geometry.KDTreeSearchParamHybrid(radius=param["normal_radius"], max_nn=30)
  pcd_source=o3d.geometry.PointCloud()
  pcd_source.points=o3d.utility.Vector3dVector(datArray[0])
  pcd_source.estimate_normals(kd_param)
  pcd_source.orient_normals_towards_camera_location()
  if 'transform' in param:
    pcd_source.transform(param['transform'])
  print("CGA solver",pcd_source)
  return [pcd_source]

def solve(datArray,param):
  pcd_target=o3d.geometry.PointCloud()
  pcd_target.points=o3d.utility.Vector3dVector(datArray[0])
  pcd_target.estimate_normals(kd_param)
  pcd_target.orient_normals_towards_camera_location()
  if 'transform' in param:
    pcd_target.transform(param['transform'])

  RT,_=cicp(pcd_source, pcd_target, param["distance_threshold"], param["icp_threshold"], 0.1)
  if 'transform' in param:
    bTc=param['transform']
    RT=np.linalg.inv(bTc).dot(RT).dot(bTc)
  result=o3d.pipelines.registration.evaluate_registration(pcd_source,pcd_target,param["icp_threshold"],RT)
  score={"transform":[RT],"fitness":[result.fitness],"rmse":[result.inlier_rmse]}
  eur=Rot.from_matrix(RT[:3,:3]).as_euler('xyz',degrees=True)
  return score
