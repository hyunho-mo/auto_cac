import argparse
import datetime
import numpy as np
import pickle
import random
import os
import pytz
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import csv
from statistics import mean, median, mode, variance
import sys
import xml.etree.ElementTree as et
import plistlib
import matplotlib.patches as patches
import bz2
import gc
import openjpeg
from label_cadidate_lesions import label_cadidate
from pydicom import dcmread
from tqdm import tqdm

## Define directory path 
current_dir = os.path.dirname(os.path.abspath(__file__))

parent_folder =  os.path.dirname(current_dir)
print ("parent_folder", parent_folder)
patch_dir_path = os.path.join(parent_folder, 'dataset/cocacoronarycalciumandchestcts-2/Gated_release_final/patch')
if not os.path.exists(patch_dir_path):
    os.makedirs(patch_dir_path)


dcm_dir_path = os.path.join(parent_folder, 'dataset/cocacoronarycalciumandchestcts-2/Gated_release_final/patient')
xml_dir_path = os.path.join(parent_folder, 'dataset/cocacoronarycalciumandchestcts-2/Gated_release_final/calcium_xml')
# heart_nii_dir_path = "/data/scratch/hmo/coca_heart"
heart_nii_dir_path = os.path.join(current_dir, "coca_nifti")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-patch_size', type=int, default=45, help="path size for FP detection")


    '''
    Data preparation step loads pids for development and test.
    Then it prepares the data for the binary classification for false positive exclusion

    Extract cadidate calcium lesions 
    - Whole heart segmentation
    - Obtain all cadidate calcium in the heart
    - Crop candidate calcium lesions (a fixed patch size)
    - Label the cadidate lesions based on the annotation(xml files)
    - Collect and save the labeled lesions

    The collected patch data are used to train the binary classifier (CNN)

    '''


    args = parser.parse_args()
    patch_size = args.patch_size

    coca_pid = os.listdir(dcm_dir_path)

    ## Check heartmask folder and heartmask nii files to check the availability of heart segmentation for the given pid
    whs_id_list = []
    heart_nii_dir_list = os.listdir(heart_nii_dir_path)
    for id in heart_nii_dir_list:
        heartmask_filename = "%s_heartmask.nii.gz" %id
        id_path = os.path.join(heart_nii_dir_path, id)
        file_path = os.path.join(id_path, heartmask_filename)
        if os.path.isfile(file_path):
            whs_id_list.append(id)


    patch_img_list = []
    patch_label_list = []

    patch_pid_list = []
    patch_slice_list = []
    patch_contour_list = []

    ## ids of xml files
    xml_files=os.listdir(xml_dir_path)
    xml_files_pids= [x.split('.')[0] for x in xml_files]

    print ("whs_id_list", whs_id_list)
    print ("xml_files_pids", xml_files_pids)

    ## Loop id existing both whs_id_list and xml filenames
    ids = list(set(whs_id_list) & set(xml_files_pids))
    ## Exclude patient having wrong annotation
    ids.remove('155')

    ## Check for only three subjects
    ids = ['196', '132', '349']
    print ("ids", ids)

    for id in tqdm(ids):
        nii_id_dir = os.path.join(heart_nii_dir_path, id)     
        raw_nii_file = os.path.join(nii_id_dir,  "%s_ct.nii.gz" %id)
        seg_nii_file = os.path.join(nii_id_dir,  "%s_heartmask.nii.gz" %id)
        dcm_folder = os.path.join(dcm_dir_path,  id)
        xml_file = os.path.join(xml_dir_path,  "%s.xml" %id)
        candidate_lesion_list, label_list, pid_list, slice_index_list, contour_index_list = label_cadidate(id, raw_nii_file, seg_nii_file, dcm_folder, xml_file, patch_size)

        # if not candidate_lesion_list:
        if not len(candidate_lesion_list)==0:

            candidate_lesion_pid_array = np.stack( candidate_lesion_list, axis=0 )
            # print ("candidate_lesion_pid_array.shape", candidate_lesion_pid_array.shape)   

            label_pid_array = np.asarray(label_list)  
            # print ("label_pid_array.shape", label_pid_array.shape)      
            patch_pid_array = np.asarray(pid_list)  
            patch_slice_array = np.asarray(slice_index_list)  
            patch_contour_array = np.asarray(contour_index_list)  

            patch_img_list.append(candidate_lesion_pid_array)
            patch_label_list.append(label_pid_array)

            patch_pid_list.append(patch_pid_array)
            patch_slice_list.append(patch_slice_array)
            patch_contour_list.append(patch_contour_array)




    print ("len(patch_img_list)", len(patch_img_list))
    print ("len(patch_label_list)", len(patch_label_list))
    ## Load raw imgs and heartmask from nii files

    ## Save to npy file
    candidate_lesion_all_array = np.concatenate( patch_img_list, axis=0 )
    label_all_array = np.concatenate( patch_label_list, axis=0 )  

    pid_all_array = np.concatenate( patch_pid_list, axis=0 )      
    slice_all_array = np.concatenate( patch_slice_list, axis=0 )      
    contour_all_array = np.concatenate( patch_contour_list, axis=0 )      

    
      

    print ("candidate_lesion_all_array.shape", candidate_lesion_all_array.shape)        
    print ("label_all_array.shape", label_all_array.shape)      



    np.save(os.path.join(patch_dir_path, "patch_img.npy"), candidate_lesion_all_array)
    np.save(os.path.join(patch_dir_path, "patch_label.npy"), label_all_array)

    np.save(os.path.join(patch_dir_path, "patch_pid.npy"), pid_all_array)
    np.save(os.path.join(patch_dir_path, "patch_slice.npy"), slice_all_array)
    np.save(os.path.join(patch_dir_path, "patch_contour.npy"), contour_all_array)

    print (label_all_array)
    print (pid_all_array)
    print (slice_all_array)
    print (contour_all_array)

    print ("number of positive samples: ", np.count_nonzero(label_all_array))
    print ("number of negative samples: ", len(label_all_array)- np.count_nonzero(label_all_array))


if __name__ == '__main__':
    main()    






