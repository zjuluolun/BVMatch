
from multiprocessing import Process, Pool
from config import root_path
import os
"""
Generate local BVFT descriptors 

Author: Luo Lun
Notes: Please change the root_path in config.py to the dataset path before using this script
"""


def generateSeqLocalDescriptors(path):
    '''
    Read clouds in the 'pointclouds' floder and save the local descriptors in  the 'local_des' folder

    @param path: the point cloud sequence path
    @return: nothing
    '''
    os.system('../build/generate_descriptors '+path+'/pointclouds/ '+ path+'/local_des/') 

if __name__ == '__main__':

    print("Please change the root_path in config.py to the dataset path before using this script")
    
    sequences = os.listdir(root_path)   
    sequences.sort()

    for seq in sequences: 
        desc_folder = root_path+'/'+seq+'/local_des/'  #mkdir local_desc folder
        if not os.path.exists(desc_folder):
            os.system('mkdir '+desc_folder)

    
    pool = Pool(4)  #thread pool
    for seq in sequences:
        pool.apply_async(generateSeqLocalDescriptors, args=(root_path+'/'+seq,))

    pool.close()  
    pool.join()  
