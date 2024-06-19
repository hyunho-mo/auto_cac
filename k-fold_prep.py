
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import sys
import os
import cv2
import matplotlib.pyplot as plt
from statistics import mean, median, mode, variance
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, recall_score
from sklearn.metrics import confusion_matrix

from label_cadidate_lesions import read_nifti_file, simple_rotate_flip

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import models.fp_classifier as fp_classifier

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

r_state = 0
random.seed(r_state)
np.random.seed(r_state)
torch.manual_seed(r_state)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Access to the server and the ESS data folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_folder =  os.path.dirname(current_dir)
patch_dir_path = os.path.join(parent_folder, 'dataset/cocacoronarycalciumandchestcts-2/Gated_release_final/patch')

dcm_dir_path = os.path.join(parent_folder, 'dataset/cocacoronarycalciumandchestcts-2/Gated_release_final/patient')
xml_dir_path = os.path.join(parent_folder, 'dataset/cocacoronarycalciumandchestcts-2/Gated_release_final/calcium_xml')
heart_nii_dir_path = "/data/scratch/hmo/coca_heart"

figure_temp_dir = os.path.join(current_dir, "fig_temp")
if not os.path.exists(figure_temp_dir):
    os.makedirs(figure_temp_dir)

figure_test_dir = os.path.join(current_dir, "fig_test")
if not os.path.exists(figure_test_dir):
    os.makedirs(figure_test_dir)

model_dir = os.path.join(current_dir, "trained_models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))


def normalize_function(volume):
    """Normalize the volume"""
    # check pixel min max first 
    # ranges from. -1024 HU to 3071 HU
    # min = -224
    # max = 1000
    min = -256
    max = 2048
    volume[volume < min] = min
    volume[volume > max] = max
    volume = ((volume - min) / (max - min))

    # threshold value 
    thres = 130
    thres_normal = ((thres - min) / (max - min))

    # use "float16" if there's out of memory issue
    volume = volume.astype("float32")
    return volume, thres_normal


def select_correct_annot_rows(w_id, patch_img_array, patch_label_array, patch_pid_array, patch_slice_array, patch_contour_array):

    correct_annot_index = np.where(patch_pid_array != w_id)
    patch_img_array = patch_img_array[correct_annot_index]
    patch_label_array = patch_label_array[correct_annot_index]
    patch_pid_array = patch_pid_array[correct_annot_index]
    patch_slice_array = patch_slice_array[correct_annot_index]
    patch_contour_array = patch_contour_array[correct_annot_index]

    return patch_img_array, patch_label_array, patch_pid_array, patch_slice_array, patch_contour_array


def oversampling_array(r_state, patch_label_array, input_array):

    index_majority_class = np.where(patch_label_array == 1)
    index_minority_class = np.where(patch_label_array == 0)

    minority_class_img = input_array[index_minority_class]
    majority_class_img = input_array[index_majority_class]

    minority_class_label = patch_label_array[index_minority_class]
    majority_class_label = patch_label_array[index_majority_class]

    # Upsample the minority class
    minority_upsampled_img = resample(minority_class_img, replace=True, n_samples=len(majority_class_img), random_state=r_state)
    minority_upsampled_label = resample(minority_class_label, replace=True, n_samples=len(majority_class_label), random_state=r_state)
    # Combine the upsampled minority class with the majority class
    balanced_array = np.concatenate([majority_class_img, minority_upsampled_img], axis=0)
    # balanced_data_label = np.concatenate([majority_class_label, minority_upsampled_label], axis=0)


    return balanced_array



def subject_split(train_subjects, patch_pid_array, patch_img_array, patch_label_array, patch_slice_array, patch_contour_array):
    train_x_list = []
    train_y_list = []
    train_pid_list = []
    train_slice_list = []
    train_contour_list = []
    for train_s in tqdm(train_subjects):
        s_index = np.where(patch_pid_array == train_s)[0]
        for ind in s_index:
            img = patch_img_array[ind]
            label = patch_label_array[ind]
            pid = patch_pid_array[ind]
            slice = patch_slice_array[ind]
            contour = patch_contour_array[ind]

            train_x_list.append(img)
            train_y_list.append(label)
            train_pid_list.append(pid)
            train_slice_list.append(slice)
            train_contour_list.append(contour)

            x_array = np.asarray(train_x_list)
            y_array = np.asarray(train_y_list)
            pid_array = np.asarray(train_pid_list)
            slice_array = np.asarray(train_slice_list)
            contour_array = np.asarray(train_contour_list)

    return x_array, y_array, pid_array, slice_array, contour_array



#function used for setting earlyStopping for CNN
class EarlyStopping:
   
    def __init__(self, patience=30, verbose=False, delta=0, path='   ', trace_func=print):
       
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = val_loss 

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss


#function used for setting earlyStopping for CNN
class EarlyStopping_accuracy:
   
    def __init__(self, patience=30, verbose=False, delta=0, path='   ', trace_func=print):
       
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_acc, model):

        score = val_acc 

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation acc increase.'''
        if self.verbose:
            self.trace_func(f'Validation acc increased ({self.val_acc_min:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_acc_min = val_acc







def main():
    '''
    Oversampling imbalanced data
    Stratified data split for train/test
    Train a CNN network
    Test the network to evaluate
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-patch_size', type=int, default=45, help="patch size")
    parser.add_argument('-sindex', help="including slice index", default=False, action="slice index as an additional channel")
    parser.add_argument('-normalize', help="normalizing pos arrays", default=False, action="normalize channel 2&3 which are for localizing")
    parser.add_argument('-oversampling', help="applying oversampling", default=False, action="not for subject level splitting")
    args = parser.parse_args()
    patch_size = args.patch_size
    oversampling = args.oversampling
    normalize = args.normalize
    sindex = args.sindex





    patch_img_array = np.load(os.path.join(patch_dir_path, "patch_img.npy"))
    patch_label_array = np.load(os.path.join(patch_dir_path, "patch_label.npy"))

    patch_pid_array = np.load(os.path.join(patch_dir_path, "patch_pid.npy"))
    patch_slice_array = np.load(os.path.join(patch_dir_path, "patch_slice.npy"))
    patch_contour_array =np.load(os.path.join(patch_dir_path, "patch_contour.npy"))


    ## Exclude wrongly annotated subjects
    # pid 417, 192
    patch_img_array, patch_label_array, patch_pid_array, patch_slice_array, patch_contour_array = select_correct_annot_rows("417", patch_img_array, patch_label_array, patch_pid_array, patch_slice_array, patch_contour_array)
    patch_img_array, patch_label_array, patch_pid_array, patch_slice_array, patch_contour_array = select_correct_annot_rows("192", patch_img_array, patch_label_array, patch_pid_array, patch_slice_array, patch_contour_array)



    if not sindex:
        patch_img_array = patch_img_array[:,:,:,[0,1,2]]

    print ("patch_img_array.shape", patch_img_array.shape)
    print ("patch_label_array.shape", patch_label_array.shape)

    index_majority_class = np.where(patch_label_array == 1)
    index_minority_class = np.where(patch_label_array == 0)

    print ("number of positive samples: ", len(index_majority_class[0]))
    print ("number of negative samples: ", len(index_minority_class[0]))

    if normalize:
        '''
        normalized x_pos, y_pos, slice_path
        '''
        pos_max = 511
        pos_min = 0

        x_pos_array = patch_img_array[:,:,:,[1]]
        patch_img_array[:,:,:,[1]] = (x_pos_array-pos_min)/(pos_max-pos_min)

        y_pos_array = patch_img_array[:,:,:,[2]]
        patch_img_array[:,:,:,[2]] = (y_pos_array-pos_min)/(pos_max-pos_min)

        if sindex:
            slice_index_array = patch_img_array[:,:,:,[3]]
            patch_img_array[:,:,:,[3]] = (slice_index_array-np.min(slice_index_array))/(np.max(slice_index_array)-np.min(slice_index_array))




    if oversampling:
        patch_label_array_cp = patch_label_array
        patch_img_array = oversampling_array(r_state, patch_label_array_cp, patch_img_array)        
        patch_pid_array = oversampling_array(r_state, patch_label_array_cp, patch_pid_array)    
        patch_slice_array = oversampling_array(r_state, patch_label_array_cp, patch_slice_array)    
        patch_contour_array = oversampling_array(r_state, patch_label_array_cp, patch_contour_array)    
        patch_label_array = oversampling_array(r_state, patch_label_array_cp, patch_label_array)    




    print ("patch_img_array.shape", patch_img_array.shape)
    print ("patch_label_array.shape", patch_label_array.shape)
    print ("patch_pid_array.shape", patch_pid_array.shape)
    print ("patch_contour_array.shape", patch_contour_array.shape)



    subjects = np.unique(patch_pid_array)
    print ("len(subjects)",  len(subjects))
    np.random.shuffle(subjects)


    subjects_all_folds = split_given_size(subjects,int(len(subjects)/5))
    subjects_all_folds = subjects_all_folds[:5]
    

    for index, test_fold in enumerate(subjects_all_folds):
        print ("index,", index)
        folds_list=[0,1,2,3,4]
        folds_list.pop(index)
        random.shuffle(folds_list)
        # print ("folds_list", folds_list)

        val_fold_index = folds_list.pop(0)
        # print ("val_fold_index", val_fold_index)
        # print ("folds_list", folds_list)


        test_subjects = test_fold
        valid_subjects = subjects_all_folds[val_fold_index]
        train_subjects = np.concatenate((subjects_all_folds[folds_list[-1]], subjects_all_folds[folds_list[-2]], subjects_all_folds[folds_list[-3]]))


        train_x, train_y, _, _, _ = subject_split(train_subjects, patch_pid_array, patch_img_array, patch_label_array, patch_slice_array, patch_contour_array)
        test_x, test_y, test_pid, test_slice, test_contour = subject_split(test_subjects, patch_pid_array, patch_img_array, patch_label_array, patch_slice_array, patch_contour_array)
        val_x, val_y, _, _, _ = subject_split(valid_subjects, patch_pid_array, patch_img_array, patch_label_array, patch_slice_array, patch_contour_array)


        np.save(os.path.join(patch_dir_path, "train_x_%s.npy" %index), train_x)
        np.save(os.path.join(patch_dir_path, "train_y_%s.npy" %index), train_y)
        np.save(os.path.join(patch_dir_path, "test_x_%s.npy" %index), test_x)
        np.save(os.path.join(patch_dir_path, "test_y_%s.npy" %index), test_y)
        np.save(os.path.join(patch_dir_path, "test_pid_%s.npy" %index), test_pid)
        np.save(os.path.join(patch_dir_path, "test_slice_%s.npy" %index), test_slice)
        np.save(os.path.join(patch_dir_path, "test_contour_%s.npy" %index), test_contour)
        np.save(os.path.join(patch_dir_path, "val_x_%s.npy" %index), val_x)
        np.save(os.path.join(patch_dir_path, "val_y_%s.npy" %index), val_y)


if __name__ == '__main__':
    main()    



