import argparse
import math
import numpy as np
#import tensorflow as tf
import socket
import importlib
import os
import sys
import pickle
import time
from sklearn.externals import joblib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
#from pointnetvlad_cls import *
#from loading_pointclouds import *
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

def get_sets_dict(filename):
	with open(filename, 'rb') as handle:
		trajectories = pickle.load(handle)
		print("Trajectories Loaded.")
		return trajectories
 

def get_hist_from_descriptors(estimator,cluster_nums,dict_to_process,src_path):#(sess, ops, dict_to_process):
    q_output = []
    
    folder = dict_to_process[1]['query']
    folder = folder[0:folder.index('_')]
    keyframe_path = src_path+folder+'/'+'local_des/'
    keyframe_names = os.listdir(keyframe_path)
    keyframe_names.sort()
    #if len(dict_to_process.keys())>len(keyframe_names)/2:
        
    for key in range(len(dict_to_process.keys())):
        print('loading '+str(key))
        idx = dict_to_process[key]['query']
        idx = int(idx[idx.index('_')+1:])
        print(keyframe_path+keyframe_names[idx])
        des = np.fromfile(keyframe_path+keyframe_names[idx],dtype=np.float32)
        des = des.reshape(des.shape[0]//216,216)
        des_tmp = np.zeros((des.shape[0],des.shape[1]))
        for i in range(36):
            des_tmp[:,i*6:i*6+6]=des[:,(36-i-1)*6:(36-i)*6]# //i*norient
        des += des_tmp
        feature = estimator.predict(des[:,0:108])
        frame_hist = np.zeros((cluster_nums,1))
        for j in range(feature.shape[0]):
            frame_hist[feature[j],0]+=1

        frame_hist=frame_hist/np.sqrt(np.sum(frame_hist**2))
        q_output.append(frame_hist)
    return q_output



def get_recall( m, n,cluster_nums):
    global DATABASE_VECTORS
    global QUERY_VECTORS

    database_output= DATABASE_VECTORS[m]
    queries_output= QUERY_VECTORS[n]

    #print(len(queries_output))
    database_nbrs = KDTree(np.array(database_output).squeeze())

    num_neighbors=25
    if num_neighbors>np.array(database_output).squeeze().shape[0]:
        num_neighbors=np.array(database_output).squeeze().shape[0]
    recall=[0]*num_neighbors

    top1_similarity_score=[]
    one_percent_retrieved=0
    threshold=max(int(round(len(database_output)/100.0)),1)

    num_evaluated=0
    for i in range(len(queries_output)):
        #print i,m,n
        true_neighbors= QUERY_SETS[n][i][m]
        if(len(true_neighbors)==0):
            continue
        num_evaluated+=1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]).reshape(1,cluster_nums),k=num_neighbors)
        #print(m,n,i,indices)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if(j==0):
                    #print(j)
                    similarity= np.dot(queries_output[i].T,database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j]+=1
                break
        #print(indices[0][0:threshold], true_neighbors)        
        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors))))>0:
            one_percent_retrieved+=1

    one_percent_recall=(one_percent_retrieved/float(num_evaluated))*100
    recall=(np.cumsum(recall)/float(num_evaluated))*100
    #print(recall)
    #print(np.mean(top1_similarity_score))
    #print(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall ,num_evaluated



from config import root_path, cluster_nums

estimator = joblib.load('./estimator.pkl')

RESULTS_FOLDER="results/"
if not os.path.exists(RESULTS_FOLDER): os.mkdir(RESULTS_FOLDER)

DATABASE_FILE= 'oxford_evaluation_database.pickle'
QUERY_FILE= 'oxford_evaluation_query.pickle'
print(DATABASE_FILE)
output_file= RESULTS_FOLDER +'results.txt'

DATABASE_SETS= get_sets_dict(DATABASE_FILE)
QUERY_SETS= get_sets_dict(QUERY_FILE)

global DATABASE_VECTORS
DATABASE_VECTORS=[]

global QUERY_VECTORS
QUERY_VECTORS=[]

recall= np.zeros(25)
count=0
similarity=[]
one_percent_recall=[]
for i in range(2):#len(DATABASE_SETS)):
    print('database',i)
    time_start = time.time()#
    DATABASE_VECTORS.append(get_hist_from_descriptors(estimator,cluster_nums,DATABASE_SETS[i],root_path))
    print(time.time()-time_start,'\n')

QUERY_VECTORS = DATABASE_VECTORS
np.save('global_descriptors.npy',DATABASE_VECTORS)  #save the global descriptors


recall_1 = []
for m in range(2):#len(QUERY_SETS)):
    for n in range(2):#len(QUERY_SETS)):
        print(m,n)
        time_start = time.time()
        if(m==n):
            continue
        pair_recall, pair_similarity, pair_opr,num_evaluated = get_recall(m, n,cluster_nums)
        recall[0:np.array(pair_recall).shape[0]]+=np.array(pair_recall)
        count+=1
        one_percent_recall.append(pair_opr)
        for x in pair_similarity:
            similarity.append(x)
        print(time.time()-time_start,'\n',pair_recall,num_evaluated)
        recall_1.append([m,n,pair_recall[0],num_evaluated])
print()
np.savetxt('recall_distribution.txt',np.array(recall_1))
ave_recall=recall/count
print(ave_recall)

#print(similarity)
average_similarity= np.mean(similarity)
print(average_similarity)

ave_one_percent_recall= np.mean(one_percent_recall)
print(ave_one_percent_recall)


#filename=RESULTS_FOLDER +'average_recall_oxford_netmax_sg(finetune_conv5).txt'
with open(output_file, "w") as output:
    output.write("Average Recall @N:\n")
    output.write(str(ave_recall))
    output.write("\n\n")
    output.write("Average Similarity:\n")
    output.write(str(average_similarity))
    output.write("\n\n")
    output.write("Average Top 1% Recall:\n")
    output.write(str(ave_one_percent_recall))
'''    
def get_recall( m, n):
    global DATABASE_VECTORS
    global QUERY_VECTORS

    database_output= DATABASE_VECTORS[m]
    queries_output= QUERY_VECTORS[n]

    print(len(queries_output))
    database_nbrs = KDTree(np.array(database_output).squeeze())

    num_neighbors=25
    recall=[0]*num_neighbors

    top1_similarity_score=[]
    one_percent_retrieved=0
    threshold=max(int(round(len(database_output)/100.0)),1)

    num_evaluated=0
    for i in range(len(queries_output)):
        print i
        true_neighbors= QUERY_SETS[n][i][m]
        if(len(true_neighbors)==0):
            continue
        num_evaluated+=1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]).reshape(1,10000),k=num_neighbors)
        print(m,n,i,indices)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if(j==0):
                    print(j)
                    similarity= np.dot(queries_output[i].T,database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j]+=1
                break
        print(indices[0][0:threshold], true_neighbors)        
        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors))))>0:
            one_percent_retrieved+=1

    one_percent_recall=(one_percent_retrieved/float(num_evaluated))*100
    recall=(np.cumsum(recall)/float(num_evaluated))*100
    #print(recall)
    #print(np.mean(top1_similarity_score))
    #print(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall 
'''
