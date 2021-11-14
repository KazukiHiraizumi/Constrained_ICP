import open3d as o3d
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat
import constrained_icp as cicp
import ransac_solver as solver
from scipy.spatial.transform import Rotation as Rot
import tflib
import yaml

voxel_size = 0.5

def main():
  with open('recipe/camera_master0.yaml') as file:
    obj = yaml.safe_load(file)
    bTc = tflib.toRTfromVec(tflib.dict2vec(obj))
  pcd = o3d.io.read_point_cloud('data/test_TZ5.ply')
  pcd_target = pcd.voxel_down_sample(voxel_size)
  pcd_source = o3d.io.read_point_cloud('recipe/surface.ply')
  pcd_target.transform(bTc)
  pcd_source.transform(bTc)

  with open('recipe/param.yaml') as file:
    param = yaml.safe_load(file)
  param=param["searcher"]

  kd_param=o3d.geometry.KDTreeSearchParamHybrid(radius=param["normal_radius"], max_nn=30)
  pcd_target.estimate_normals(search_param=kd_param)
  pcd_target.orient_normals_towards_camera_location()
  pcd_source.estimate_normals(search_param=kd_param)
  pcd_source.orient_normals_towards_camera_location()

  zaxis=o3d.geometry.TriangleMesh.create_cylinder(radius=5, height=600)
  zaxis.transform(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
  o3d.visualization.draw_geometries([pcd_target,pcd_source,zaxis])

  transform_ini=np.eye(4)
  RT,_=cicp.cicp(pcd_source, pcd_target, transform_ini, param["distance_threshold"], param["icp_threshold"])
  result=o3d.pipelines.registration.evaluate_registration(pcd_source,pcd_target,param["icp_threshold"],RT)
  print("RT=",RT)
  eur=Rot.from_matrix(RT[:3,:3]).as_euler('xyz',degrees=True)
  print("Trans=",RT[:3,3])
  print("Euler angle=",eur)
  print("RMSE",result.inlier_rmse)
  print("Fitness",result.fitness)

if __name__ == '__main__':
    main()
