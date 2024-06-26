import nibabel as nib
from scipy import ndimage
from pydicom import dcmread, read_file
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
from lib import process_xml
import models.fp_classifier as fp_classifier
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))


figure_coca_dir = os.path.join(current_dir, 'Figure_interpret')
if not os.path.exists(figure_coca_dir):
    os.makedirs(figure_coca_dir)

# figure_patch_dir = os.path.join(current_dir, 'Figure_patch')
# if not os.path.exists(figure_patch_dir):
#     os.makedirs(figure_patch_dir)

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def normalize(volume):
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


def transform_to_hu(medical_image, image):
    # medical_image = pydicom.read_file
    # image = medical_image.pixel_array

    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    # hu_image = hu_image.astype(np.int16)

    return hu_image


def simple_rotate_flip(img):
    img = ndimage.rotate(img, -90, reshape=False)
    img = np.flip(img, axis=1)
    return img

## Function for load nii file
def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    # scan = scan.get_fdata()
    scan = scan.get_data()

    return scan

# Read dcm files and assign image to data["G"] dictionary
def read_dcm_folder(dcm_folder):
    sdir = os.listdir(dcm_folder)
    sdir_name = sdir[0]
    sdir_path = os.path.join(dcm_folder, sdir_name)
    files = os.listdir(sdir_path)
    image_list = []
    for filename in sorted(files, reverse=True):
    # for filename in sorted(files, reverse=False):
        # print ("filename", filename)
        filepath = os.path.join(sdir_path, filename)
        image_index = filename.split(".")[0].split("-")[-1]
        if filepath.endswith(".dcm"):

            # image = dcmread(filepath).pixel_array  
            # image_list.append(image)

            medical_image = read_file(filepath)
            image = medical_image.pixel_array
            hu_image = transform_to_hu(medical_image, image)
            image_list.append(hu_image)

    return image_list



def protocol_thresholding_manual (heart_seg_images, thres_mask, thres):

    '''
    Extract pixels above threshold of 130HU and 1mm^2
    '''

    true_positions = list(zip(*np.where(thres_mask == True)))
    for pos in true_positions:

        if (pos[0] >= 510) or (pos[1] >= 510):
            pass
        else:
            adjacent_checker = heart_seg_images[pos[0]-1:pos[0]+1, pos[1]-1 : pos[1]+1] >= thres
            adjacent_checker_list = adjacent_checker.flatten().tolist()

            left_checker = heart_seg_images[pos[0]-2 : pos[0], pos[1]-1 : pos[1]+1] >= thres
            left_checker_list = left_checker.flatten().tolist()

            right_checker = heart_seg_images[pos[0] :pos[0]+2, pos[1]-1 : pos[1]+1] >= thres
            right_checker_list = right_checker.flatten().tolist()

            up_checker = heart_seg_images[pos[0]-1 : pos[0]+1, pos[1] : pos[1]+2] >= thres
            up_checker_list = up_checker.flatten().tolist()

            down_checker = heart_seg_images[pos[0]-1 : pos[0]+1, pos[1]-2 : pos[1]] >= thres
            down_checker_list = down_checker.flatten().tolist()
    
            if (all(adjacent_checker_list) == True) or (all(left_checker_list) == True) or (all(right_checker_list) == True) or (all(up_checker_list) == True) or (all(down_checker_list) == True):
                pass
            else:
                thres_mask[pos] = 0

    thres_images =  heart_seg_images * thres_mask
    # print ("thres_images.shape", thres_images.shape)

    return thres_images, thres_mask


def compute_scores(pid, raw_nii_file, seg_nii_file, dcm_folder, xml_file, patch_size, pth_filepath):

    '''
    Loop each slice and find cadidate lesions in each slice (using cv2.findContours())
    Extract patches based on thresholding and label each patch based on annotation(xml)
    '''
    print ("pid", pid)
    pos_max = 511
    pos_min = 0

    figure_pid_dir = os.path.join(figure_coca_dir, pid)
    if not os.path.exists(figure_pid_dir):
        os.makedirs(figure_pid_dir)

    ## Loop slices
    raw_images = read_nifti_file(raw_nii_file)
    heart_masks = read_nifti_file(seg_nii_file)

    ## Load fp detector (trained torch model) 
    if torch.cuda.is_available():
        model = torch.load(pth_filepath)
    else:
        model = torch.load(pth_filepath, map_location=torch.device('cpu'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




    ## Rotate nifti images
    raw_images = simple_rotate_flip(raw_images)
    
    heart_masks = simple_rotate_flip(heart_masks)


    raw_images_wo_norm = raw_images
    raw_images, norm_thres = normalize(raw_images)

    # ## dcm read
    # image_list = read_dcm_folder(dcm_folder)
    # dcm_image_array = np.asarray(image_list)
    # dcm_image_array, norm_thres = normalize(dcm_image_array)

    ## Load annotation from xml file
    # The function returns dict with 
    # key: integer representing ImageIndex, 
    # its values is a python list comprising "cid" (artery name) and "pixels" (pix coor of the lesion)
    num_slices = raw_images.shape[2]
    annot_patient = process_xml(xml_file)

    # print ("annot_patient", annot_patient)


    ## Loop slices containing annotated lesions
    ## Check all the keys (image index) of annot_patient  
    annot_img_index_list = list(annot_patient.keys())

    patch_list = []
    label_list = []

    pid_list = []
    slice_index_list = []
    contour_index_list = []

    pid_annot_lesion_count = 0

    ref_agatston_score = 0
    ref_vol_score = 0
    computed_agatston_score = 0
    computed_vol_score = 0



    for xml_image_index in annot_img_index_list:
        slice_idx = xml_image_index
        patch_slice_list = []
        label_slice_list = []

        image_index = num_slices - xml_image_index
        # print ("image_index", image_index)

        ## Apply heart segmentation mask
        # first_nii_img = raw_images[:,:,image_index]
        first_nii_img = raw_images[:,:, -image_index]
        first_nii_img_wo_norm = raw_images_wo_norm[:,:, -image_index]
        first_heart_mask = heart_masks[:,:, -image_index]
        # first_dcm_img = image_list[image_index]


        ## Apply heart seg mask
        first_heart_mask = morphology.binary_erosion(first_heart_mask, iterations=10)
        masked_first_nii_img = first_nii_img * first_heart_mask


        fig = plt.figure()
        plt.imshow(masked_first_nii_img, cmap=plt.cm.gray)
        # plt.imshow(masked_image, cmap="gray", vmin=0)
        plt.savefig(os.path.join(figure_pid_dir, '%s_nifti_heart.png' %(slice_idx)))


        ## Thresholding 
        calc_mask = first_heart_mask > norm_thres
        calc_masked_first_nii_img, calc_mask = protocol_thresholding_manual (masked_first_nii_img, calc_mask, norm_thres)

        calc_masked_first_nii_img_raw = calc_masked_first_nii_img.copy()



        ## Assign annotation for the current slice
        # current_slice_index = num_slices - image_index
        current_slice_index = xml_image_index
        print ("current_slice_index", current_slice_index)
        print ("image_index", image_index)

        # key: integer representing ImageIndex, 
        # its values is a python list comprising dict contaning "cid" (artery name) and "pixels" (pix coor of the lesion)
        annot_slice = annot_patient[current_slice_index]
        print ("number of lesions in annotation: ", len(annot_slice))
        pid_annot_lesion_count += len(annot_slice)


        annot_px_list = []
        annot_px_tuple = ()
        for l_id, annot_lesion in enumerate(annot_slice):
            lesion_pixels = annot_lesion['pixels']
            lesion_pixels_array = np.array(lesion_pixels,dtype="int32")
            lesion_pixels_array = np.reshape(lesion_pixels_array, (lesion_pixels_array.shape[0], 1, lesion_pixels_array.shape[1]))
            annot_px_list.append(lesion_pixels_array)

        annot_contours = tuple(annot_px_list)


        gray_three = cv2.merge([first_nii_img,first_nii_img,first_nii_img])
        with_annot_contours = cv2.drawContours(gray_three, annot_contours, -1 ,(255,0,0), 1)

        fig = plt.figure()
        # plt.imshow(first_nii_img, cmap="gray", vmin=0, vmax=1)
        plt.imshow(with_annot_contours)
        plt.savefig(os.path.join(figure_pid_dir, '%s_annot_contours.png' %(slice_idx)))
        plt.close()

        calc_annot_pos_all_list = []
        for ac_index in range(len(annot_contours)):

            mask = np.zeros(first_nii_img.shape, np.uint8)
            mask = cv2.merge([mask,mask,mask])

            cv2.drawContours(image=mask, contours=annot_contours, contourIdx=ac_index, color=(0,255,255), thickness=cv2.FILLED)
            # fig = plt.figure()                
            # plt.imshow(mask)
            # plt.savefig(os.path.join(figure_dir, '%s_%s_filled_contour_annot_%s.png' %(pid, image_index, ac_index)))

            mask_array = np.sum(mask, axis=2)
            calc_candid_pos = np.stack(np.nonzero(mask_array), axis=-1)
            calc_annot_pos_list = []
            calc_pos_y = calc_candid_pos[:,0]
            calc_pos_x = calc_candid_pos[:,1]
            for y, x in zip(calc_pos_y, calc_pos_x):
                calc_annot_pos_list.append((x,y))

            calc_annot_pos_all_list.append(calc_annot_pos_list)



            area_px = len(calc_candid_pos)
            area_mm = area_px*(180/512)*(180/512)  


            hu_values = []
            for y, x in zip(calc_pos_y, calc_pos_x):

                hu_value = int(first_nii_img_wo_norm[(y,x)]) 
                hu_values.append(hu_value)

            # hu_values = [calc_masked_first_nii_img[px] for px in calc_candid_pos_list]
            # print ("hu_values", hu_values)
            max_hu = max(hu_values)
            # print ("max_hu", max_hu)

            if 130 <= max_hu <= 199:
                hu_factor = 1
            elif 200 <= max_hu <= 299:
                hu_factor = 2
            elif 300 <= max_hu <= 399:
                hu_factor = 3
            elif 400 <= max_hu:
                hu_factor = 4                        
            # Factor for agatston
            # print ("hu_factor", hu_factor)
            
            ref_contour_agatston_score = area_mm * hu_factor

            ref_agatston_score += ref_contour_agatston_score
            ref_vol_score += area_mm * 3

            print ("ref_contour_agatston_score", ref_contour_agatston_score)
            print ("area_mm * 3", area_mm * 3)
    #######################################################################################
       
       
        ## Edge detection and findcontours
        # Find the contours on the inverted binary image, and store them in a list
        # Contours are drawn around white blobs.
        # hierarchy variable contains info on the relationship between the contours
        contours, hierarchy = cv2.findContours(calc_mask.astype(np.uint8),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)



        ## Translate contours
        contours_trans_list = []
        for index, c in enumerate(contours):
            contours_trans_list.append(c - 1)

        contours = tuple(contours_trans_list)



        # if len(contours) >= 10 :
        #     px_thres = 36
        # else:
        #     px_thres = 16

        px_thres = 6

        gray_three = cv2.merge([first_nii_img,first_nii_img,first_nii_img])
        with_contours = cv2.drawContours(gray_three, contours, -1 ,(255,0,0), 1)


        ### Loop for contours ###
        for index, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            # # Make sure contour area is large enough
            # cv2.rectangle(with_contours,(x,y), (x+w,y+h), (255,255,0), 0)

            x_center = x + int(w/2)
            y_center = y + int(h/2)

            new_x = x_center - int(patch_size/2)
            new_y = y_center - int(patch_size/2)
        
            pos_match_counter = 0         

            img_patch = first_nii_img[new_y:new_y+patch_size, new_x:new_x+patch_size]
            ## Prepare 2d patch conveying spational info in y-direction 
            y_pos_vector = np.arange(new_y, new_y+patch_size)
            # print ("y_pos_vector", y_pos_vector)
            y_pos_patch = np.tile(y_pos_vector, (patch_size, 1))
            y_pos_patch = np.transpose(y_pos_patch)
            # print ("y_pos_patch", y_pos_patch)

            ## Prepare 2d patch conveying spational info in x-direction 
            x_pos_vector = np.arange(new_x, new_x+patch_size)
            x_pos_patch = np.tile(x_pos_vector, (patch_size, 1))
            # print ("x_pos_patch", x_pos_patch)

            ## Normalize x, y pos patch
            x_pos_patch = (x_pos_patch-pos_min)/(pos_max-pos_min)
            y_pos_patch = (y_pos_patch-pos_min)/(pos_max-pos_min)



            if img_patch.shape[0] == 0:
                continue       
            else:
                
                ## neglects contours on the corners 
                if (img_patch.shape[0] != 45) or (img_patch.shape[1] != 45):
                    print ("neglect contours on the corners")
                    continue


                patch = np.dstack((img_patch, x_pos_patch, y_pos_patch))        

            ## Check annotations and compare it for labeling the patch
            # Retrieve pixel positions of the pixels in the contour
            mask = np.zeros(first_nii_img.shape, np.uint8)
            mask = cv2.merge([mask,mask,mask])
            # cv2.drawContours(mask, annot_contours, -1 ,(255,0,0), thickness=cv2.FILLED)
            # cv2.drawContours(mask, annot_contours, -1 ,(255,0,0), 1)

            mask2 = np.zeros(first_nii_img.shape, np.uint8)
            mask2 = cv2.merge([mask2,mask2,mask2])

            cv2.drawContours(image=mask, contours=contours, contourIdx=index, color=(0,255,255), thickness=cv2.FILLED)
            # cv2.drawContours(image=mask, contours=contours, contourIdx=index, color=(0,255,255), thickness=-1)

            cv2.drawContours(image=mask2, contours=contours, contourIdx=index, color=(0,255,255), thickness=1)

            # fig = plt.figure()
            # # plt.imshow(masked_first_nii_img, cmap=plt.cm.gray)
            # plt.imshow(mask2, cmap="gray", vmin=0)
            # # plt.savefig(os.path.join(figure_pid_dir, '%s_%s_nifti_calc.png' %(id_dir, slide_ref)))
            # plt.savefig(os.path.join(figure_pid_dir, '%s_%s_mask2_%s.png' %(id_dir, slice_idx, index)))


            mask_array = np.sum(mask, axis=2)
            mask2_array = np.sum(mask2, axis=2)

            calc_candid_pos = np.stack(np.nonzero(mask_array), axis=-1)
            calc_candid_pos = unique_rows(calc_candid_pos)
            calc_contour_pos = np.stack(np.nonzero(mask2_array), axis=-1)
            calc_contour_pos = unique_rows(calc_contour_pos)
            calc_neglect_pos = np.concatenate((calc_candid_pos, calc_contour_pos))
            calc_neglect_pos = unique_rows(calc_neglect_pos)




            # print ("calc_candid_pos_list", calc_candid_pos_list)
            # print ("num of px: ", len(calc_candid_pos_list))

            if len(calc_candid_pos) <= px_thres:
                print ("too small lension")
                label = 0
                # print ("calc_candid_pos", calc_candid_pos)
                calc_masked_first_nii_img[calc_neglect_pos] = 0
                lesion_size = "small"
                continue

            else:
                lesion_size = "large"
                ## check the output of DL-based fp detector when the size of lesion is larger than 6 pixels        
                model.eval()
                with torch.no_grad():
                    patch_4d = np.expand_dims(patch, axis=0)
                    # print ("patch_4d.shape", patch_4d.shape)
                    img = np.transpose(patch_4d, axes=[0, 3, 2, 1])
                    img = torch.Tensor(img)
                    # print ("img.shape", img.shape)
                    img = img.to(device)
                    output = model(img)
                    exp = torch.exp(output).cpu()
                    exp_sum = torch.sum(exp, dim=1) 
                    softmax = exp/exp_sum.unsqueeze(-1)
                    prob = list(softmax.detach().cpu().numpy())
                    predictions = np.argmax(prob, axis=1)

                label = int(predictions.item())

                if label == 1: 
                    ## Calculate area of this contour

                    # area_px = cv2.contourArea(c)
                    ## Number of pixels
                    # area_px = len(calc_neglect_pos)
                    area_px = len(calc_candid_pos)

                    # print ("area_px", area_px)
                    area_mm = area_px*(180/512)*(180/512)  
                    # area_mm = area_px*(1/3)*(1/3)  

                    # print ("area_mm", area_mm)
                    ## max HU value within contour
                    # print ("calc_masked_first_nii_img.shape", calc_masked_first_nii_img.shape)
                    # print ("calc_neglect_pos.shape", calc_neglect_pos.shape)

                    calc_pos_y = calc_candid_pos[:,0]
                    calc_pos_x = calc_candid_pos[:,1]
                    # calc_pos_y = calc_neglect_pos[:,0]
                    # calc_pos_x = calc_neglect_pos[:,1]
                    hu_values = []
                    for y, x in zip(calc_pos_y, calc_pos_x):

                        hu_value = int(first_nii_img_wo_norm[(y,x)]) 
                        hu_values.append(hu_value)

                    # hu_values = [calc_masked_first_nii_img[px] for px in calc_candid_pos_list]
                    # print ("hu_values", hu_values)
                    max_hu = max(hu_values)
                    # print ("max_hu", max_hu)

                    if 130 <= max_hu <= 199:
                        hu_factor = 1
                    elif 200 <= max_hu <= 299:
                        hu_factor = 2
                    elif 300 <= max_hu <= 399:
                        hu_factor = 3
                    elif 400 <= max_hu:
                        hu_factor = 4                        
                    # Factor for agatston
                    # print ("hu_factor", hu_factor)
                    
                    computed_contour_agatston_score = area_mm * hu_factor

                    computed_agatston_score += computed_contour_agatston_score
                    computed_vol_score += area_mm * 3
                    # computed_vol_score += area_mm * 2

                    print ("computed_contour_agatston_score", computed_contour_agatston_score)
                    print ("area_mm * 3, ", area_mm * 3)

                    cv2.rectangle(with_contours, (new_x,new_y), (new_x+patch_size,new_y+patch_size), (0, 128, 0), 1)

                elif label == 0:
                    cv2.rectangle(with_contours, (new_x,new_y), (new_x+patch_size,new_y+patch_size), (255, 255, 0), 0)

        fig = plt.figure()                
        plt.imshow(with_contours)
        plt.savefig(os.path.join(figure_pid_dir, '%s_patch.png' %(slice_idx)))
        plt.close()


    # return patch_list, label_list, pid_list, slice_index_list, contour_index_list

    ref_agatston_score = round(ref_agatston_score, 1)
    ref_vol_score = round(ref_vol_score, 1)

    computed_agatston_score = round(computed_agatston_score, 1)
    computed_vol_score = round(computed_vol_score, 1)

    print ("ref_agatston_score", ref_agatston_score)
    print ("ref_vol_score", ref_vol_score)
    print ("computed_agatston_score", computed_agatston_score)
    print ("computed_vol_score", computed_vol_score)

    return ref_agatston_score, ref_vol_score, computed_agatston_score, computed_vol_score
