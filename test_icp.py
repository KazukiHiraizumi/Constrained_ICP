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
  pcd = o3d.io.read_point_cloud('data/test_RZ3.ply')
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

if __name__ == '__main__':
    main()
