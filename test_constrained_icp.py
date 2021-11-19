import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import tflib
import yaml
try:
  from rovi_utils import cga_solver as solver
except:
  import cga_solver as solver

voxel_size = 0.5

def main():
  with open('recipe/camera_master0.yaml') as file:
    obj = yaml.safe_load(file)
    bTc = tflib.toRTfromVec(tflib.dict2vec(obj))
  pcd = o3d.io.read_point_cloud('data/test_RZ3.ply')
  pcd_target = pcd.voxel_down_sample(voxel_size)
  pcd_source = o3d.io.read_point_cloud('recipe/surface.ply')
  pcd_target.transform(bTc)
  pcd_source.transform(bTc)
#  zaxis=o3d.geometry.TriangleMesh.create_cylinder(radius=5, height=600)
#  o3d.visualization.draw_geometries([pcd_target,pcd_source,zaxis])

  with open('recipe/param.yaml') as file:
    param = yaml.safe_load(file)
  param=param["searcher"]
#  param["transform"]=bTc

  source_pnts=np.array(pcd_source.points)
  solver.learn([source_pnts],param)
  target_pnts=np.array(pcd_target.points)
  result=solver.solve([target_pnts],param)
  RT=result["transform"][0]
  print("RT=",RT)
  eur=Rot.from_matrix(RT[:3,:3]).as_euler('xyz',degrees=True)
  print("Trans=",RT[:3,3])
  print("Euler angle=",eur)
  print("RMSE",result["rmse"][0])
  print("Fitness",result["fitness"][0])

if __name__ == '__main__':
    main()
