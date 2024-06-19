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
from label_cadidate_scoring import compute_scores
from pydicom import dcmread
from tqdm import tqdm
import pandas as pd


## Access to the server and the ESS data folder
current_dir = os.path.dirname(os.path.abspath(__file__))

# tabular_dir = os.path.join(current_dir, 'Tabular')
# if not os.path.exists(tabular_dir):
#     os.makedirs(tabular_dir)

parent_folder =  os.path.dirname(current_dir)
print ("parent_folder", parent_folder)
patch_dir_path = os.path.join(parent_folder, 'dataset/cocacoronarycalciumandchestcts-2/Gated_release_final/patch')
if not os.path.exists(patch_dir_path):
    os.makedirs(patch_dir_path)


dcm_dir_path = os.path.join(parent_folder, 'dataset/cocacoronarycalciumandchestcts-2/Gated_release_final/patient')
xml_dir_path = os.path.join(parent_folder, 'dataset/cocacoronarycalciumandchestcts-2/Gated_release_final/calcium_xml')
heart_nii_dir_path = "/data/scratch/hmo/coca_heart"

model_dir = os.path.join(current_dir, 'trained_models')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-patch_size', type=int, default=45, help="patch size")
    parser.add_argument('-trained_model', type=str, default='fp_vgg_trained_model_all.pth', help="pth filename of the trained model")



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
    pth_filename = args.trained_model


    pth_filepath = os.path.join(model_dir, pth_filename)



    df = pd.read_csv("coca_val_ids.csv", sep = ",")
    coca_pid = df['val_ids'].astype(str)

    ## Check heartmask folder and heartmask nii files to check the availability of heart segmentation for the given pid
    whs_id_list = []
    heart_nii_dir_list = os.listdir(heart_nii_dir_path)
    for id in heart_nii_dir_list:
        heartmask_filename = "%s_heartmask.nii.gz" %id
        id_path = os.path.join(heart_nii_dir_path, id)
        file_path = os.path.join(id_path, heartmask_filename)
        if os.path.isfile(file_path):
            whs_id_list.append(id)

    # print ("whs_id_list", whs_id_list)
    # print ("len(whs_id_list)", len(whs_id_list))

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
    ids = list(set(whs_id_list) & set(xml_files_pids) & set(coca_pid))    

    print ("ids", ids)
    print ("len(ids)", len(ids))


    ids_list =[]
    ref_agatston_list = []
    ref_volume_list = []
    computed_agatston_list = []
    computed_volume_list = []

    for id in tqdm(ids):
        nii_id_dir = os.path.join(heart_nii_dir_path, id)     
        raw_nii_file = os.path.join(nii_id_dir,  "%s_ct.nii.gz" %id)
        seg_nii_file = os.path.join(nii_id_dir,  "%s_heartmask.nii.gz" %id)
        dcm_folder = os.path.join(dcm_dir_path,  id)
        xml_file = os.path.join(xml_dir_path,  "%s.xml" %id)
        ref_agatston, ref_volume, computed_agatston, computed_volume = compute_scores(id, raw_nii_file, seg_nii_file, dcm_folder, xml_file, patch_size, pth_filepath)

        ids_list.append(id)
        ref_agatston_list.append(ref_agatston)
        ref_volume_list.append(ref_volume)
        computed_agatston_list.append(computed_agatston)
        computed_volume_list.append(computed_volume)



    df = pd.DataFrame([])
    df["patient_name"] = ids_list
    df["given_agatston"] = ref_agatston_list
    df["computed_agatston"] = computed_agatston_list
    df["given_vol"] = ref_volume_list
    df["computed_vol"] = computed_volume_list

    df.to_csv("coca_score_comparison.csv", index=False)

if __name__ == '__main__':
    main()    






