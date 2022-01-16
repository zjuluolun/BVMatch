# -*- coding=utf-8 -*-
"""
Train the bag-of-words model 

Author: Luo Lun

"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.externals import joblib
import sys

from config import root_path, cluster_nums


sequences = os.listdir(root_path)

sequences.sort()
descriptors_train = []
for seq in sequences:

  if '2014' in seq:  #using the sequences collected in 2014 for training
    print(seq)
    keyframe_names = os.listdir(root_path+seq+'/'+'local_des/')
    keyframe_names.sort()
    for i in range(len(keyframe_names)):
          
        des = np.fromfile(root_path+seq+'/'+'local_des/'+keyframe_names[i],dtype=np.float32)
        des = des.reshape(des.shape[0]//216,216)
        des_tmp = np.zeros((des.shape[0],des.shape[1]))
        for i in range(36):
            des_tmp[:,i*6:i*6+6]=des[:,(36-i-1)*6:(36-i)*6]
        des += des_tmp         #rotation invariance
        descriptors_train.append(des[:,0:108])


descriptors_train = np.vstack(descriptors_train)
print("num of desc: ", descriptors_train.shape )

estimator = MiniBatchKMeans(n_clusters=cluster_nums,batch_size=cluster_nums)
estimator.fit(descriptors_train)
print('finish training '+str(cluster_nums)+" clusters")
joblib.dump(estimator,'./estimator.pkl')

