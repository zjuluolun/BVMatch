"""
Define the dataset path the cluster nums of the bag-of-words model

Author: Luo Lun

"""

root_path="../data/OxfordRobotCar/" 
"""
root_path is the path to the dataset, the dataset directory should be orgnized like this

root_path:
 - seq1/
   - pointcloud_locations_20m.csv
   - pointclouds/
   - local_des/ #this folder will be made after generating the local descriptors

 - seq2
   - pointcloud_locations_20m.csv
   - pointclouds/
   - local_des/ #this folder will be made after generating the local descriptors
 
 ......

"""

cluster_nums = 10000   #cluster nums (or number of words)