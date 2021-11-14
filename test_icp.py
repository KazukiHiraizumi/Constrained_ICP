import open3d as o3d
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat
import ransac_solver as solver
from scipy.spatial.transform import Rotation as Rot
import tflib
import yaml

voxel_size=0.5

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
  solver.learn([np.array(pcd_source.points)],param)
  result=solver.solve([np.array(pcd_target.points)],param)
  RT=np.array(result["transform"][0])

  eur=Rot.from_matrix(RT[:3,:3]).as_euler('xyz',degrees=True)
  print("Trans=",RT[:3,3])
  print("Euler angle=",eur)
  print("RMSE",result["rmse"])

  Tm = np.eye(4)
  Tm[:3,:3]=Rot.from_euler('xyz',[-eur[0],-eur[1],0],degrees=True).as_matrix()
  Tm=Tm.dot(RT)
  pcd_source.transform(Tm)
  param["eval_threshold"]=param["icp_threshold"]
  param["icp_threshold"]=0
  param["distance_threshold"]=0
  solver.clear(param)
  solver.learn([np.array(pcd_source.points)],param)
  result=solver.solve([np.array(pcd_target.points)],param)
  RT=np.array(result["transform"][0])
  eur=Rot.from_matrix(RT[:3,:3]).as_euler('xyz',degrees=True)
  print("Trans=",RT[:3,3])
  print("Euler angle=",eur)
  print("RMSE",result["rmse"])
  print("Fitness",result["fitness"])

if __name__ == '__main__':
    main()
