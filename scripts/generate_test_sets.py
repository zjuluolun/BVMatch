# -*-coding=utf-8 -*-

"""
Generate the pickle files of the test set

Author: Luo Lun

"""
import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import random
import sys

##########################################

def output_to_file(output, filename):
	with open(filename, 'wb') as handle:
	    pickle.dump(output, handle, protocol=2)#pickle.HIGHEST_PROTOCOL)
	print("Done ", filename)


def construct_query_and_database_sets(base_path, folders, pointcloud_fols, filename, output_name):
	database_trees=[]
	test_trees=[]
	database_list=[]
	for folder in folders:
		#print(folder)
		df_database= pd.DataFrame(columns=['file','northing','easting'])
		df_test= pd.DataFrame(columns=['file','northing','easting'])
		
		df_locations= pd.read_csv(os.path.join(base_path, folder,filename),sep=',')
	
		database_rows=[]
		for index, row in df_locations.iterrows():
			row['file'] = folder+'_'+str(index)	
			#print(row['file'][0:row['file'].index('_')],row['file'][row['file'].index('_')+1:])
			#entire business district is in the test set
			if(output_name=="business"):
				df_test=df_test.append(row, ignore_index=True)
			elif(1):#check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
				df_test=df_test.append(row, ignore_index=True)
			df_database=df_database.append(row, ignore_index=True)
			database_rows.append([row['northing'], row['easting']])
		print('database',folder,len(database_rows))
		database_tree = KDTree(df_database[['northing','easting']])
		database_list.append(database_rows)
		test_tree = KDTree(df_test[['northing','easting']])
		database_trees.append(database_tree)
		test_trees.append(test_tree)
                
	test_sets=[]
	database_sets=[]
	for folder in folders:
		database={}
		test={} 
		df_locations= pd.read_csv(os.path.join(base_path,folder,filename),sep=',')
		df_locations['timestamp']= folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
		df_locations=df_locations.rename(columns={'timestamp':'file'})
		for index,row in df_locations.iterrows():
			row['file'] = folder+'_'+str(index)			
			#entire business district is in the test set
			if(output_name=="business"):
				test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
			elif(1):#check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
				test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
			database[len(database.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
		database_sets.append(database)
		test_sets.append(test)		
	for i in range(0,len(database_sets)):
		tree=database_trees[i]
		print(len(test_sets[i].keys()))
		for j in range(len(test_sets)):#range(0,1):#:
			if(i==j):
				continue
			for key in range(len(test_sets[j].keys())):
			    #print()
				coor=np.array([[test_sets[j][key]["northing"],test_sets[j][key]["easting"]]])
				index = tree.query_radius(coor, r=25)
				index_brute=[]
				for k in range(len(database_list[i])):
					row = database_list[i][k]
					if np.sqrt((row[0]-coor[0,0])**2+(row[1]-coor[0,1])**2)<25:
						index_brute.append(k)
       
				test_sets[j][key][i]=index[0].tolist()

	output_to_file(database_sets,  output_name+'_evaluation_database.pickle')
	output_to_file(test_sets,  output_name+'_evaluation_query.pickle')
	return test_sets

###Building database and query files for evaluation
from config import root_path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


folders=[]

             
sequences=sorted(os.listdir(os.path.join(BASE_DIR,root_path)))

for seq  in sequences:#len(all_folders)):
    if '2015' in seq:# and all_folders[i] not in short:
    		folders.append(seq)
#folders=folders[0:3]
print(folders)
construct_query_and_database_sets(root_path, folders, "/pointclouds/", "pointcloud_locations_20m.csv", "oxford")
