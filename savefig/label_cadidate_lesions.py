'''
Python file including function for FP detection and visualization
'''

import nibabel as nib
from scipy import ndimage
from pydicom import dcmread, read_file
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
from lib import process_xml



current_dir = os.path.dirname(os.path.abspath(__file__))

figure_coca_dir = os.path.join(current_dir, 'Figure_coca')
if not os.path.exists(figure_coca_dir):
    os.makedirs(figure_coca_dir)

# figure_patch_dir = os.path.join(current_dir, 'Figure_patch')
# if not os.path.exists(figure_patch_dir):
#     os.makedirs(figure_patch_dir)

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
    """Transfrom pixel value of medical image to HU"""
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


def label_cadidate(pid, raw_nii_file, seg_nii_file, dcm_folder, xml_file, patch_size):

    '''
    Loop each slice and find cadidate lesions in each slice (using cv2.findContours())
    Extract patches based on thresholding and label each patch based on annotation(xml)
    '''
    print ("pid", pid)

    figure_pid_dir = os.path.join(figure_coca_dir, pid)
    if not os.path.exists(figure_pid_dir):
        os.makedirs(figure_pid_dir)

    ## Loop slices
    raw_images = read_nifti_file(raw_nii_file)
    heart_masks = read_nifti_file(seg_nii_file)

    ## Rotate nifti images
    raw_images = simple_rotate_flip(raw_images)
    heart_masks = simple_rotate_flip(heart_masks)


    raw_images_wo_norm = raw_images
    raw_images, norm_thres = normalize(raw_images)

    ## dcm read
    image_list = read_dcm_folder(dcm_folder)
    dcm_image_array = np.asarray(image_list)
    dcm_image_array, norm_thres = normalize(dcm_image_array)

    ## Load annotation from xml file
    # The function returns dict with 
    # key: integer representing ImageIndex, 
    # its values is a python list comprising "cid" (artery name) and "pixels" (pix coor of the lesion)
    num_slices = raw_images.shape[2]
    annot_patient = process_xml(xml_file)

    ## Loop slices containing annotated lesions
    ## Check all the keys (image index) of annot_patient  
    annot_img_index_list = list(annot_patient.keys())
    
    patch_list = []
    label_list = []
    pid_list = []
    slice_index_list = []
    contour_index_list = []
    pid_annot_lesion_count = 0

    for xml_image_index in annot_img_index_list:

        patch_slice_list = []
        label_slice_list = []

        image_index = num_slices - xml_image_index
        # print ("image_index", image_index)

        ## Apply heart segmentation mask
        first_nii_img = raw_images[:,:, -image_index]
        first_nii_img_wo_norm = raw_images_wo_norm[:,:, -image_index]
        first_heart_mask = heart_masks[:,:, -image_index]
        first_dcm_img = image_list[image_index]

        fig = plt.figure()
        plt.imshow(first_nii_img, cmap=plt.cm.gray)
        plt.savefig(os.path.join(figure_pid_dir, '%s_%s_nifti.png' %(pid, image_index)))
        plt.close()

        ## Apply heart seg mask
        first_heart_mask = morphology.binary_erosion(first_heart_mask, iterations=10)
        masked_first_nii_img = first_nii_img * first_heart_mask

        fig = plt.figure()
        # plt.imshow(masked_first_nii_img, cmap=plt.cm.gray)
        plt.imshow(masked_first_nii_img, cmap="gray", vmin=0)
        plt.savefig(os.path.join(figure_pid_dir, '%s_%s_nifti_heart.png' %(pid, image_index)))
        plt.close()

        ## Thresholding 
        calc_mask = first_heart_mask > norm_thres
        calc_masked_first_nii_img, calc_mask = protocol_thresholding_manual (masked_first_nii_img, calc_mask, norm_thres)

        fig = plt.figure()
        # plt.imshow(masked_first_nii_img, cmap=plt.cm.gray)
        plt.imshow(calc_masked_first_nii_img, cmap="gray", vmin=0)
        plt.savefig(os.path.join(figure_pid_dir, '%s_%s_nifti_calc.png' %(pid, image_index)))
        plt.close()


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


        # Draw the contours (in red) on the original image and display the result
        # Input color code is in BGR (blue, green, red) format
        # -1 means to draw all contours    
        # gray_three = cv2.merge([first_nii_img_wo_norm,first_nii_img_wo_norm,first_nii_img_wo_norm])
        gray_three = cv2.merge([first_nii_img,first_nii_img,first_nii_img])
        with_contours = cv2.drawContours(gray_three, contours, -1 ,(255,0,0), 1)

        fig = plt.figure()
        # plt.imshow(first_nii_img, cmap="gray", vmin=0, vmax=1)
        plt.imshow(with_contours)
        plt.savefig(os.path.join(figure_pid_dir, '%s_%s_contours.png' %(pid, image_index)))
        plt.close()


        ## Assign annotation for the current slice
        # current_slice_index = num_slices - image_index
        current_slice_index = xml_image_index

        # key: integer representing ImageIndex, 
        # its values is a python list comprising dict contaning "cid" (artery name) and "pixels" (pix coor of the lesion)
        annot_slice = annot_patient[current_slice_index]
        print ("number of lesions in annotation: ", len(annot_slice))
        pid_annot_lesion_count += len(annot_slice)

        annot_px_list = []
        annot_px_tuple = ()
        ## Convert the type of annotated lesions to numpy array and append to python list
        for l_id, annot_lesion in enumerate(annot_slice):
            lesion_pixels = annot_lesion['pixels']
            lesion_pixels_array = np.array(lesion_pixels,dtype="int32")
            lesion_pixels_array = np.reshape(lesion_pixels_array, (lesion_pixels_array.shape[0], 1, lesion_pixels_array.shape[1]))
            annot_px_list.append(lesion_pixels_array)

        annot_contours = tuple(annot_px_list) # match to the cv2 contour format
        gray_img = cv2.merge([first_nii_img,first_nii_img,first_nii_img]) # 3 channels for plotting
        with_annot_contours = cv2.drawContours(gray_img, annot_contours, -1 ,(255,0,0), 1)

        fig = plt.figure()
        # plt.imshow(first_nii_img, cmap="gray", vmin=0, vmax=1)
        plt.imshow(with_annot_contours)
        plt.savefig(os.path.join(figure_pid_dir, '%s_%s_annot_contours.png' %(pid, image_index)))
        plt.close()

        calc_annot_pos_all_list = []
        for ac_index in range(len(annot_contours)):

            mask = np.zeros(first_nii_img.shape, np.uint8)
            mask = cv2.merge([mask,mask,mask])

            # Fill the annotated contours
            cv2.drawContours(image=mask, contours=annot_contours, contourIdx=ac_index, color=(0,255,255), thickness=cv2.FILLED)
            fig = plt.figure()                
            plt.imshow(mask)
            plt.savefig(os.path.join(figure_pid_dir, '%s_%s_filled_contour_annot_%s.png' %(pid, image_index, ac_index)))
            plt.close()

            mask_array = np.sum(mask, axis=2)
            calc_candid_pos = np.stack(np.nonzero(mask_array), axis=-1)
            calc_annot_pos_list = []
            calc_pos_y = calc_candid_pos[:,0]
            calc_pos_x = calc_candid_pos[:,1]
            for y, x in zip(calc_pos_y, calc_pos_x):
                calc_annot_pos_list.append((x,y))

            calc_annot_pos_all_list.append(calc_annot_pos_list)

        if len(contours) == 0:
            print ("zero contour found")

        ## analyze and extract patch based on slice after applying threshold
        for index, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            ## Make sure contour area is large enough
            # cv2.rectangle(with_contours,(x,y), (x+w,y+h), (255,255,0), 0)

            x_center = x + int(w/2)
            y_center = y + int(h/2)
            new_x = x_center - int(patch_size/2)
            new_y = y_center - int(patch_size/2)

            ## Crop image based on it coordinates in opencv
            ## cv2.rectangle(rightImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
            ## cropImg=rightImg[y:y+h,x:x+w]
            img_patch = first_nii_img[new_y:new_y+patch_size, new_x:new_x+patch_size]

            ## Prepare 2d patch conveying spational info in y-direction 
            y_pos_vector = np.arange(new_y, new_y+patch_size)
            y_pos_patch = np.tile(y_pos_vector, (patch_size, 1))
            y_pos_patch = np.transpose(y_pos_patch)

            ## Prepare 2d patch conveying spational info in x-direction 
            x_pos_vector = np.arange(new_x, new_x+patch_size)
            x_pos_patch = np.tile(x_pos_vector, (patch_size, 1))

            ## Prepare 2d patch conveying the slice index
            sindex_vector = np.ones((patch_size,), dtype=int)          
            sindex_vector = sindex_vector * image_index  
            sindex_patch = np.tile(sindex_vector, (patch_size, 1))

            ## Stack patch image and x,y spational information. Each of them processed in the different channel of the network.
            patch = np.dstack((img_patch, x_pos_patch, y_pos_patch, sindex_patch))        

            ## Check annotations and compare it for labeling the patch
            # Retrieve pixel positions of the pixels in the contour
            mask = np.zeros(first_nii_img.shape, np.uint8)
            mask = cv2.merge([mask,mask,mask])


            cv2.drawContours(image=mask, contours=contours, contourIdx=index, color=(0,255,255), thickness=cv2.FILLED)

            mask_array = np.sum(mask, axis=2)

            calc_candid_pos = np.stack(np.nonzero(mask_array), axis=-1)
            calc_candid_pos_list = []
            calc_pos_y = calc_candid_pos[:,0]
            calc_pos_x = calc_candid_pos[:,1]
            for y, x in zip(calc_pos_y, calc_pos_x):
                # calc_candid_pos_list.append((x,y))
                calc_candid_pos_list.append((x,y))



            if len(calc_candid_pos_list) <= 6:
                print ("too small lension")
                continue

            else:
                ## check overrapping pixel position with the pixels in the annotated lesions                
                ## append patch only if the size of lesion is larger than 6 pixels      

                label = 0
                for calc_annot_pos in calc_annot_pos_all_list:
                    # print ("calc_annot_pos", calc_annot_pos)
                    
                    ## If elements of calc_candid_pos_list are duplicated with any of calc_annot_pos_all_list, then label the patch as 1
                    px_duplicate = len(set(calc_candid_pos_list) & set(calc_annot_pos)) 
                    # print ("px_duplicate", px_duplicate)
                    if px_duplicate >= 3:
                        label = 1

                if label == 1:
                    cv2.rectangle(with_contours, (new_x,new_y), (new_x+patch_size,new_y+patch_size), (0, 128, 0), 0)
                else:
                    cv2.rectangle(with_contours, (new_x,new_y), (new_x+patch_size,new_y+patch_size), (255, 255, 0), 0)

                patch_list.append(patch)
                label_list.append(label)

                pid_list.append(pid)
                slice_index_list.append(image_index)
                contour_index_list.append(index)

                patch_slice_list.append(patch)
                label_slice_list.append(label)


        fig = plt.figure()                
        plt.imshow(with_contours)
        plt.savefig(os.path.join(figure_pid_dir, '%s_%s_patch.png' %(pid, image_index)))
        plt.close()

        for index, patch in enumerate(patch_slice_list):
            img_patch = patch[:,:,0]
            plt.imshow(img_patch, cmap=plt.cm.gray)
            plt.savefig(os.path.join(figure_pid_dir, '%s_%s_patch_%s_label_%s.png' %(pid, image_index, index, label_slice_list[index])))
            plt.close()



    return patch_list, label_list, pid_list, slice_index_list, contour_index_list
