
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

def print_evaluation(lst):
    
    lst_mean = np.mean(lst)
    lst_std = np.std(lst)
    print (lst)
    print ("mean +- std %s +- %s" %(round(lst_mean, 4), round(lst_std, 4)))

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
    parser.add_argument('-batch_size', type=int, default=64, help="batch size")
    parser.add_argument('-n_epochs', type=int, default=30, help="number of epochs")
    parser.add_argument('-lr', type=float, default=0.0001, help="learning rate")

    args = parser.parse_args()
    patch_size = args.patch_size
    batch_size = args.batch_size


    # number of epochs to train the model
    n_epochs = args.n_epochs
    learning_rate = args.lr


    folds_list=[0,1,2,3,4]

    acc_list =[]
    auc_list =[]
    sensitivity_list =[]
    specificity_list =[]

    for f_index in folds_list:
        print ("f_index", f_index)

        train_x = np.load(os.path.join(patch_dir_path, "train_x_%s.npy" %f_index))
        train_y = np.load(os.path.join(patch_dir_path, "train_y_%s.npy" %f_index) )
        test_x = np.load(os.path.join(patch_dir_path, "test_x_%s.npy" %f_index) )
        test_y = np.load(os.path.join(patch_dir_path, "test_y_%s.npy" %f_index) )
        test_pid = np.load(os.path.join(patch_dir_path, "test_pid_%s.npy" %f_index) )
        test_slice = np.load(os.path.join(patch_dir_path, "test_slice_%s.npy" %f_index) )
        test_contour = np.load(os.path.join(patch_dir_path, "test_contour_%s.npy" %f_index) )
        val_x = np.load(os.path.join(patch_dir_path, "val_x_%s.npy" %f_index) )
        val_y = np.load(os.path.join(patch_dir_path, "val_y_%s.npy" %f_index) )


        train_x = np.transpose(train_x, axes=[0, 3, 2, 1])
        val_x = np.transpose(val_x, axes=[0, 3, 2, 1])
        test_x = np.transpose(test_x, axes=[0, 3, 2, 1])

        print ("train_x.shape", train_x.shape)

        # Stratified data split train/validation
        tensor_x = torch.Tensor(train_x) # transform to torch tensor
        tensor_y = torch.Tensor(train_y)

        print ("tensor_x.shape", tensor_x.shape)

        train_dataset = TensorDataset(tensor_x,tensor_y) 
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)


        print ("test_pid", test_pid)
        print ("test_slice", test_slice)
        print ("test_pid.shape", test_pid.shape)
        print ("test_slice.shape", test_slice.shape)
        print ("test_x.shape", test_x.shape)

        # train_batch = next(iter(train_dataloader))
        # for index, patch in enumerate(train_batch[0]):
        #     for cindex in range(len(patch)):
        #         patch_img = patch[cindex]
        #         label = train_batch[1][index]

        #         fig = plt.figure()
        #         plt.imshow(patch_img, cmap=plt.cm.gray)
        #         plt.savefig(os.path.join(figure_temp_dir, '%s_patch_train_batch_%s, %s.png' %(index, cindex, label)))



        tensor_x = torch.Tensor(val_x) # transform to torch tensor
        tensor_y = torch.Tensor(val_y)
        val_dataset = TensorDataset(tensor_x,tensor_y) 
        val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size) 

        tensor_x = torch.Tensor(test_x) # transform to torch tensor
        tensor_y = torch.Tensor(test_y)
        test_dataset = TensorDataset(tensor_x,tensor_y) 
        test_dataloader = DataLoader(test_dataset) 

        train_data_size = len(train_dataset)
        val_data_size = len(val_dataset)
        test_data_size = len(test_dataset)   

        print ("train_data_size", train_data_size)

        # Model load and parameter settings        
        model = fp_classifier.VGG_Net(in_channel = train_x.shape[1])
        optimizer = Adam(model.parameters(), lr=learning_rate)
        criterion = CrossEntropyLoss()

        early_stopping = EarlyStopping(patience=30, verbose = True, path=os.path.join(model_dir, 'fp_vgg_trained_model_%s.pth' %f_index))
            

            

        # # checking if GPU is available
        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()
            model = model.to(device)
            criterion = criterion.to(device)




        # # num of testing steps
        total_test_step = 0
        logs = []
        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        val_acc_list = []
        # Train and validate the model
        for i in range(n_epochs):    

            print("--------start training epoch: {}-------------------".format(i + 1))
            # # start of training
            model.train()
            total_test_loss = 0
            total_trainaccuracy = 0
            total_testaccuracy = 0
            total_val_loss = 0
            total_train_loss = 0
            
            total_train_step = 0
            
            train_total_tp = 0
            val_total_tp = 0
            for data in train_dataloader:
                imgs, targets = data

                imgs = imgs.to(device)
                targets = targets.type(torch.LongTensor) 
                targets = targets.to(device)
                outputs = model(imgs)


                softmax = torch.exp(outputs).cpu()
                prob = list(softmax.detach().cpu().numpy())
                

                exp = torch.exp(outputs).cpu()
                exp_sum = torch.sum(exp, dim=1) 
                softmax = exp/exp_sum.unsqueeze(-1)
                
                prob = list(softmax.detach().cpu().numpy())
                

                predictions = np.argmax(prob, axis=1)
                targets_array = targets.detach().cpu().numpy().astype(int)
                tp_count = (predictions == targets_array).sum()
                train_total_tp += tp_count

                loss = criterion(outputs, targets)
                total_train_loss = total_train_loss + loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_step += 1
                if total_train_step % 100 == 0:
                    print("num of training:{},loss:{}".format(total_train_step, loss.item()))
            
            # # start of testing
            model.eval()
            with torch.no_grad():
                for data in val_dataloader:
                    imgs, targets = data
                    imgs = imgs.to(device)
                    targets = targets.type(torch.LongTensor) 
                    targets = targets.to(device)
                    outputs = model(imgs)

                    exp = torch.exp(outputs).cpu()
                    exp_sum = torch.sum(exp, dim=1) 
                    softmax = exp/exp_sum.unsqueeze(-1)
                    prob = list(softmax.detach().cpu().numpy())
                    

                    predictions = np.argmax(prob, axis=1)
                    targets_array = targets.detach().cpu().numpy().astype(int)
                    tp_count = (predictions == targets_array).sum()
                    val_total_tp += tp_count


                    val_loss = criterion(outputs, targets)
                    total_val_loss = total_val_loss + val_loss.item()


            # # #if the loss in validation set do not decrease, then stop training
            # early_stopping(total_test_loss, model)
            # #if the accuracy in validation set do not increase, then stop training
            # # early_stopping(total_testaccuracy, model)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
                    
            train_acc = train_total_tp / train_data_size
            val_acc = val_total_tp / val_data_size
            print("train accuracy: {}".format(train_acc))
            print("val accuracy: {}".format(val_acc))

            print("total_train_loss: {}".format(total_train_loss))
            print("total_val_loss: {}".format(total_val_loss))

            train_loss_list.append(total_train_loss)
            val_loss_list.append(total_val_loss)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)

            early_stopping(total_val_loss, model)
            # # if the accuracy in validation set do not increase, then stop training
            # early_stopping(total_testaccuracy, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break



        train_log={"train_acc": train_acc_list, "val_acc": val_acc_list, "train_loss": train_loss_list, "val_loss": val_loss_list}

        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        ax = ax.ravel()

        for i, metric in enumerate(["acc", "loss"]):
            ax[i].plot(train_log["train_" + metric])
            ax[i].plot(train_log["val_" + metric])
            ax[i].set_title("Model {}".format(metric), fontsize=16)
            ax[i].set_xlabel("epochs", fontsize=16)
            ax[i].set_ylabel(metric, fontsize=16)

            ax[i].tick_params(axis="x", labelsize=14)
            ax[i].tick_params(axis="y", labelsize=14)

            ax[i].legend(["train", "val"], fontsize=16)

        plt.savefig(os.path.join(figure_temp_dir, 'fp_vgg_training_curves_%s.png' %f_index ), bbox_inches='tight')
        plt.savefig(os.path.join(figure_temp_dir, 'fp_vgg_training_curves_%s.eps' %f_index ), bbox_inches='tight')
        plt.close()

        # Test the model
        test_total_tp = 0
        model = torch.load(os.path.join(model_dir, 'fp_vgg_trained_model_%s.pth' %f_index))
        model.eval()

        output_list = []
        label_list = []
        pred_class_list =[]
        test_pid_list =[]
        test_slice_list =[]
        test_contour_list =[]
        test_x_list =[]
        test_y_list =[]

        with torch.no_grad():
            for index, data in enumerate(test_dataloader):
                imgs, targets = data
                imgs_cpu = imgs
                imgs = imgs.to(device)
                targets = targets.type(torch.LongTensor) 
                targets = targets.to(device)
                outputs = model(imgs)

                exp = torch.exp(outputs).cpu()
                exp_sum = torch.sum(exp, dim=1) 
                softmax = exp/exp_sum.unsqueeze(-1)

                softmax = torch.exp(outputs).cpu()
                prob = list(softmax.detach().cpu().numpy())




                predictions = np.argmax(prob, axis=1)
                targets_array = targets.detach().cpu().numpy().astype(int)
                # print ("predictions", predictions)
                # print ("targets_array", targets_array)

                output_list.append(prob[0][1])
                label_list.append(targets_array[0])
                pred_class_list.append(predictions[0])

                tp_count = (predictions == targets_array).sum()            
                test_total_tp += tp_count

                test_pid_list.append(test_pid[index])
                test_slice_list.append(test_slice[index])
                test_contour_list.append(test_contour[index])

                # print ("imgs_cpu.shape", imgs_cpu.shape)
                test_img_x = imgs_cpu.numpy()[0][1]*511
                # print ("test_img_x", test_img_x)
                test_img_y = imgs_cpu.numpy()[0][2]*511
                # print ("test_img_y", test_img_y)           

                test_x_list.append(int(test_img_x[0][0]))
                test_y_list.append(int(test_img_y[0][0]))

            test_acc = test_total_tp / test_data_size


        # Plot ROC curve and confusion matrix
        y_test = label_list
        test_result = pred_class_list

        prediction_results = output_list


        # print ("test_pid_list", test_pid_list)
        # print ("test_slice_list", test_slice_list)
        # print ("test_x_list", test_x_list)
        # print ("test_y_list", test_y_list)

        # print ("y_test", y_test)
        # print ("test_result", test_result)
        # print ("prediction_results", prediction_results)

        fpr, tpr, thresholds = roc_curve(y_test, test_result, pos_label=2)
        fpr, tpr, thresholds = roc_curve(y_test, prediction_results)
        print ("fpr", fpr)
        print ("tpr", tpr)
        print ("thresholds", thresholds)
        auc_value = auc(fpr, tpr)
        print ("auc_value", auc_value)
        print ("test_acc", test_acc)


        # method I: plt
        plt.figure(figsize=(4,4))
        # plt.title('Receiver Operating Characteristic, VGG', fontsize=16)
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % auc_value)
        plt.legend(loc = 'lower right', fontsize=15)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate', fontsize=15)
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_temp_dir, 'ROC_curve_vgg_%s_%s.png' %(len(y_test), f_index)), bbox_inches='tight')
        plt.savefig(os.path.join(figure_temp_dir, 'ROC_curve_vgg_%s_%s.eps' %(len(y_test), f_index)), bbox_inches='tight')
        plt.close()

        tn, fp, fn, tp = confusion_matrix(y_test, test_result).ravel()
        specificity = tn / (tn+fp)
        print ("specificity", specificity)

        sensitivity = tp / (tp + fn) 
        print ("sensitivity", sensitivity)

        sensitivity = recall_score(y_test , test_result)
        print ("sensitivity", sensitivity)


        cm = confusion_matrix(y_test, test_result)
        # Plot confusion matrix
        plt.figure(figsize=(4,4.5))
        plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
        # plt.title('Confusion matrix - FP detection, VGG', size = 15)
        # plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["False", "True"], rotation=0, size = 15)
        plt.yticks(tick_marks, ["False", "True"], rotation=90, size = 15)
        plt.tight_layout()
        plt.ylabel('Actual label', size = 15)
        plt.xlabel('Predicted label', size = 15)
        plt.colorbar(fraction=0.0458, pad=0.04)
        width, height = cm.shape
        for x in range(width):
            for y in range(height):
                plt.annotate(str(cm[x][y]), xy=(y, x), 
                horizontalalignment='center',
                verticalalignment='center', fontsize=15)
        plt.savefig(os.path.join(figure_temp_dir, 'Conf_matrix_fp_vgg_%s_%s.png' %(len(y_test),  f_index)), bbox_inches='tight')
        plt.savefig(os.path.join(figure_temp_dir, 'Conf_matrix_fp_vgg_%s_%s.eps' %(len(y_test),  f_index)), bbox_inches='tight')
        plt.close()

        # df = pd.DataFrame([])
        # df["pid"] = test_pid_list
        # df["slice"] = test_slice_list
        # df["contour"] = test_contour_list
        # df["new_x"] = test_x_list
        # df["new_y"] = test_y_list
        # df["pred_class"] = pred_class_list
        # df["label"] = label_list


        print ("test_total_tp", test_total_tp)
        print ("test_data_size", test_data_size)
        print ("auc_value", auc_value)
        print ("test_acc", test_acc)
        print ("specificity", specificity)
        print ("sensitivity", sensitivity)

        index_majority_class = np.where(test_y == 1)
        index_minority_class = np.where(test_y == 0)

        print ("number of test positive samples: ", len(index_majority_class[0]))
        print ("number of test negative samples: ", len(index_minority_class[0]))

        acc_list.append(test_acc)
        auc_list.append(auc_value)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)



    print ("acc")
    print_evaluation(acc_list)

    print ("auc")
    print_evaluation(auc_list)

    print ("sensitivity_list")
    print_evaluation(sensitivity_list)

    print ("specificity_list")
    print_evaluation(specificity_list)


if __name__ == '__main__':
    main()    



