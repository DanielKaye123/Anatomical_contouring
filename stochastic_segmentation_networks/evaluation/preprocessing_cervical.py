import os
import SimpleITK as sitk
import argparse
import pandas as pd
import numpy as np

import nibabel as nib
import shutil

train_ids = ['zzAMLART001',
            'zzAMLART002',
            'zzAMLART003',
            'zzAMLART004',
            'zzAMLART005',
            'zzAMLART006',
            'zzAMLART007',
            'zzAMLART008',
            'zzAMLART009',
            'zzAMLART010',
            'zzAMLART011',
            'zzAMLART012',
            'zzAMLART013',
            'zzAMLART014',
            'zzAMLART015',
            'zzAMLART016',
            'zzAMLART017',
            'zzAMLART018',
            'zzAMLART019',
            'zzAMLART020',
            'zzAMLART021',
            'zzAMLART022',
            'zzAMLART023',
            'zzAMLART024',
            'zzAMLART025',
            'zzAMLART026',
            'zzAMLART027',
            'zzAMLART028',
            'zzAMLART029',
            'zzAMLART030',
            'zzAMLART031',
            'zzAMLART032',
            'zzAMLART033',
            'zzAMLART034',
            'zzAMLART035',
            'zzAMLART036',
            'zzAMLART037',
            'zzAMLART038',
            'zzAMLART039',
            'zzAMLART040',
            'zzAMLART041',
            'zzAMLART042',
            'zzAMLART043',
            'zzAMLART044',
            'zzAMLART045',
            'zzAMLART046',
            'zzAMLART047',
            'zzAMLART048',
            'zzAMLART049',
            'zzAMLART050',
            'zzAMLART051',
            'zzAMLART052',
            'zzAMLART053',
            'zzAMLART054',
            'zzAMLART055',
            'zzAMLART056',
            'zzAMLART057',
            'zzAMLART058',
            'zzAMLART059',
            'zzAMLART060',
            'zzAMLART061',
            'zzAMLART062',
            'zzAMLART063',
            'zzAMLART064',
            'zzAMLART065',
            'zzAMLART066',
            'zzAMLART067',
            'zzAMLART068',
            'zzAMLART069',
            'zzAMLART070']

            
valid_ids = ['zzAMLART071',
            'zzAMLART072',
            'zzAMLART073',
            'zzAMLART074',
            'zzAMLART075',
            'zzAMLART076',
            'zzAMLART077',
            'zzAMLART078',
            'zzAMLART079',
            'zzAMLART080']

test_ids = ['zzAMLART081',
            'zzAMLART082',
            'zzAMLART083',
            'zzAMLART084',
            'zzAMLART085',
            'zzAMLART086',
            'zzAMLART087',
            'zzAMLART088',
            'zzAMLART089',
            'zzAMLART090',
            'zzAMLART091',
            'zzAMLART092',
            'zzAMLART093',
            'zzAMLART094',
            'zzAMLART095',
            'zzAMLART096',
            'zzAMLART097',
            'zzAMLART098',
            'zzAMLART099',
            'zzAMLART100']


hold_out_ids = ['zzAMLIOV001', 
                'zzAMLIOV002',
                'zzAMLIOV003',
                'zzAMLIOV004',
                'zzAMLIOV005',
                'zzAMLIOV006',
                'zzAMLIOV007',
                'zzAMLIOV008',
                'zzAMLIOV009',
                'zzAMLIOV010'
                ]




# Creates a label map based on the input masks
def create_segmentation_label_map(id_):
    # Create segmentation label map
        label_map = np.array([], dtype=np.dtype('u1') )
        for (i,mask) in enumerate(masks):
            mask_path = parse_args.input_dir + "/"+ id_ + "/mask_" + mask + ".nii.gz"
            try:
                img = nib.load(mask_path)
                img_array = np.array(img.dataobj)
                if(len(label_map) == 0):
                    img_header= img.header.copy()
                    label_map = np.zeros(img_array.shape, dtype=np.dtype('u1') )
            
                # Assumes no overlapping labels
                label_map[img_array != 0] = i+1 

                ni_img = nib.Nifti1Image(label_map, None, header= img_header)
                output_path = os.path.join(parse_args.output_dir, id_) + f'_seg.nii.gz'
                nib.save(ni_img, output_path)
                output_dataframe.loc[id_, 'seg'] = output_path
            except Exception as e: 
                print(e)
                print("Failed", id_, mask_path)
                pass
        #print(label_map[label_map > 0])
        
# Gets body mask that Kat has drawn. This is used in the final version
def get_body_mask(id_):
    body_mask  = sitk.ReadImage(os.path.join(parse_args.input_dir, id_) + '/mask_BODY.nii.gz')
    output_path = os.path.join(parse_args.output_dir, id_) + f'_body_mask.nii.gz'
    output_dataframe.loc[id_, 'sampling_mask'] = output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(body_mask, output_path)
    return body_mask

# Generates new body mask based on threshold intensities. Not used in the final version. 
def create_body_mask(id_):
    # Load the CT image
    img = sitk.ReadImage(os.path.join(parse_args.input_dir, id_) + '/image.nii.gz')
    data = sitk.GetArrayFromImage(img)

    # Define the CT body range
    body_min = -100  # Minimum CT value for the body
    body_max = 300   # Maximum CT value for the body

    # Create the body mask based on the CT body range
    body_mask = np.logical_and(data >= body_min, data <= body_max).astype(np.uint8)

    # Create a SimpleITK image from the body mask
    body_mask_img = sitk.GetImageFromArray(body_mask)
    body_mask_img.CopyInformation(img)
    body_mask_img = sitk.Cast(body_mask_img, sitk.sitkUInt8)

  
    output_path = os.path.join(parse_args.output_dir, id_) + f'_body_mask.nii.gz'
    output_dataframe.loc[id_, 'sampling_mask'] = output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(body_mask_img, output_path)

# Helper funciton
# Z-score normalisation prevents non-invertible cov but not ideal for CT. So don't use.
def z_score_normalisation(channel, brain_mask, cutoff_percentiles=(5., 95.), cutoff_below_mean=True):
    low, high = np.percentile(channel[brain_mask.astype(bool)], cutoff_percentiles)
    norm_mask = np.logical_and(brain_mask, np.logical_and(channel > low, channel < high))
    if cutoff_below_mean:
        norm_mask = np.logical_and(norm_mask, channel > np.mean(channel))
    masked_channel = channel[norm_mask]
    normalised_channel = (channel - np.mean(masked_channel)) / np.std(masked_channel)
    return normalised_channel

# Main. But don't use Z-score
def z_normalise_input_img(id_, body_mask):
    #Normalise main input image
    suffix = "CT"
    channel = sitk.ReadImage(os.path.join(parse_args.input_dir, id_) + '/image.nii.gz')
    channel_array = sitk.GetArrayFromImage(channel)
    normalised_channel_array = z_score_normalisation(channel_array, sitk.GetArrayFromImage(body_mask))
    normalised_channel = sitk.GetImageFromArray(normalised_channel_array) #sitk.GetImageFromArray(channel_array)
    normalised_channel.CopyInformation(channel)
    output_path = os.path.join(parse_args.output_dir, id_) + f'_{suffix:s}.nii.gz'
    output_dataframe.loc[id_, suffix] = output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(normalised_channel, output_path)

## Helper function -> call global_normalize_input_img
# def range_normalization(img, min, max):
#     # img_min = -1000    # Global min across all images is 1,000
#     # img_max = 29014   # Global max across all images is 29014.0

#     img_min = min
#     img_max = max
#     img_range = img_max - img_min
#     img_normalized = (2 * (img - img_min) / img_range) - 1
#     return img_normalized

def preserve_intensities(image, min_value, max_value):
    # Create a mask for out-of-bounds intensities
    mask = np.logical_or(image < min_value, image > max_value)
    
    # Clip the intensities within the specified range
    clipped_image = np.clip(image, min_value, max_value)
    
    # Preserve original intensities for out-of-bounds values
    preserved_image = np.where(mask, image, clipped_image)
    
    return preserved_image


def range_normalization(ct_image, min_range, max_range):
      # Convert CT image to numpy array
    #ct_array = sitk.GetArrayFromImage(ct_image)
    ct_array = ct_image

    # Mask for pixels within the specified range
    mask = np.logical_and(ct_array >= min_range, ct_array <= max_range)

    # Create a copy of the array and set out-of-range values to the minimum value
    normalized_array = np.copy(ct_array)
    #normalized_array[~mask] = min_range

    # Normalize within the specified range
    normalized_array = (normalized_array - min_range) / (max_range - min_range)

    # Apply the scaling to match the desired output range
    output_min = -3
    output_max = 3
    normalized_array = normalized_array * (output_max - output_min) + output_min

    # Preserve original intensities for out-of-bounds values
    preserved_image = np.where(mask, normalized_array, ct_array)

    return preserved_image

# This uses linear interpolation method to map ranges. E.g from -100,800 to -1,1
def linear_interpolation(ct_scan, input_min, input_max, output_min, output_max):
    # Clip intensities to have the same min/max range
    clipped_ct_scan = np.clip(ct_scan, input_min, input_max)
    
    # Compute the slope and intercept for linear interpolation
    slope = (output_max - output_min) / (input_max - input_min)
    intercept = output_min - slope * input_min
    
    # Apply linear interpolation and normalization
    normalized_ct_scan = slope * clipped_ct_scan + intercept
    
    return normalized_ct_scan


def range_normalize_input_img(id_, input_min, input_max):
    
    # Normalise main input image
    suffix = "CT"
    if not hold_out:
        channel = sitk.ReadImage(os.path.join(parse_args.input_dir, id_) + '/image.nii.gz')
    else:
        id_ = id_.rsplit(".nii.gz", 1)[0]
        channel = sitk.ReadImage(os.path.join(parse_args.input_dir, id_))
    channel_array = sitk.GetArrayFromImage(channel)
    # body_mask = sitk.ReadImage(os.path.join(parse_args.input_dir, id_) + '/body.nii.gz')
   # body_mask_array = sitk.GetArrayFromImage(body_mask)
    #masked_channel_array = np.where(body_mask_array > 0, channel_array, -1000)
    normalised_channel_array = linear_interpolation(channel_array, input_min, input_max, -1,1)
    normalised_channel = sitk.GetImageFromArray(normalised_channel_array)
    normalised_channel.CopyInformation(channel)
    if not hold_out:
        output_path = os.path.join(parse_args.output_dir, id_) + f'_{suffix:s}.nii.gz'
    else:
        output_path = os.path.join(parse_args.output_dir, id_) + f'_{suffix:s}.nii.gz'
    output_dataframe.loc[id_, suffix] = output_path
    print(output_dataframe)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(normalised_channel, output_path)



## Clips before shifting the mean (Not good)
def old_map_to_range_mean_0(ct_scan, input_min, input_max):
    #mean_intensity = 103 # Mean intensity between -100,800 after it has been shifted to 0 900
    mean_intensity = 100 # Mean intensity between -100,200 after it has been shifted to 0, 300
    #ct_scan = np.clip(ct_scan, input_min, input_max)

    ct_scan = np.clip(ct_scan, input_min, input_max)
    ct_scan = ct_scan - input_min - mean_intensity + (input_max - input_min) /2

    

    ct_scan = ct_scan / ((input_max - input_min) / 2)

    ct_scan = ct_scan - 1
    return ct_scan


# Shifts the intensities to have mean of 0 across the dataset. Clips to -1,1 range after the mean has been shifted.
# This normalisation works / prevents non-invertible matrix
def map_to_range_mean_0(ct_scan, input_min, input_max):
    mean_intensity = 103 # Mean intensity between -100,800 after it has been shifted to 0 900
    #mean_intensity = 100 # Mean intensity between -100,200 after it has been shifted to 0, 300
    #ct_scan = np.clip(ct_scan, input_min, input_max)

    
    ct_scan = ct_scan - input_min - mean_intensity + (input_max - input_min) /2


    ct_scan = ct_scan / ((input_max - input_min) / 2)

    ct_scan = ct_scan - 1

    ct_scan = np.clip(ct_scan, -1, 1)
    return ct_scan


# Main -> zero mean a dataset and map between range e.g -100,800, -1,1
def range_0_mean_input_img(id_, input_min, input_max):
    # Normalise main input image
    suffix = "CT"
    if not hold_out:
    
        channel = sitk.ReadImage(os.path.join(parse_args.input_dir, id_) + '/image.nii.gz')
    else:
        channel = sitk.ReadImage(os.path.join(parse_args.input_dir, id_) + '.nii.gz')
    channel_array = sitk.GetArrayFromImage(channel)
    # body_mask = sitk.ReadImage(os.path.join(parse_args.input_dir, id_) + '/body.nii.gz')
    #body_mask_array = sitk.GetArrayFromImage(body_mask)
    #masked_channel_array = np.where(body_mask_array > 0, channel_array, -1000)
    normalised_channel_array = map_to_range_mean_0(channel_array, input_min, input_max)
    normalised_channel = sitk.GetImageFromArray(normalised_channel_array)
    normalised_channel.CopyInformation(channel)
    output_path = os.path.join(parse_args.output_dir, id_) + f'_{suffix:s}.nii.gz'
    output_dataframe.loc[id_, suffix] = output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(normalised_channel, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',
                        required=True,
                        type=str,
                        help='Path to input directory.')
    parser.add_argument('--output-dir',
                        required=True,
                        type=str,
                        help='Path to output directory.')
    parser.add_argument('--task-id',
                        required=True,
                        type=str,
                        help='Name of Task. Used to create asset folder.')
    
    hold_out = False
    parse_args, unknown = parser.parse_known_args()
    output_dataframe = pd.DataFrame()


    ## NOTE: Final working setup used was for task 7. Which is a linear interpolation -100,800 to -1,1.
    # But range 0 mean also seems to work well

    #parse_args.input_dir = "/vol/biomedic3/bglocker/radiotherapy/kat100/nifti"
    norm_type = ""
    task_id = int(parse_args.task_id)
    if  task_id < 4:
        masks = ["Bones", "FemoralHead_L", "FemoralHead_R", "Bladder", "Anorectum", "Bowel-bag", "Bowel-loops", "CTVp"]
        norm_type = "z"
    elif task_id == 4:
        masks = [ "Bladder", "Anorectum", "CTVp"]
        norm_type = "global"
    elif task_id == 5:
        masks = [ "Bladder", "Anorectum", "CTVp"]
        norm_type = "range"
        min = -100
        max = 800
    elif task_id == 6:
        masks = ["CTVn", "CTVp", "Anorectum"]
        norm_type = "range"
        min = -200
        max = 200
    elif task_id == 7:
        masks = ["CTVn", "CTVp", "Anorectum"]
        norm_type = "range"
        min = -100
        max = 800
    elif task_id == 20:
        masks = ["CTVn", "CTVp", "Anorectum"]
        norm_type = "range_0_mean"
        min = -100
        max = 800
    elif task_id == 40:
        masks = ["CTVn"]
        norm_type = "range_0_mean"
        min = -100
        max = 800
    elif task_id == 50:
        masks = ["CTVn", "CTVp", "Anorectum"]
        norm_type = "range_0_mean"
        min = -100
        max = 200
    elif task_id == 60:
        masks = ["CTVn", "CTVp", "Anorectum"]
        norm_type = "range_0_mean"
        min = -100
        max = 800
    elif task_id == 100:
        # Hold out test
        masks = ["CTVn", "CTVp", "Anorectum"]
        norm_type = "range"
        min = -100
        max = 800
        hold_out = True

    


    for id_ in os.listdir(parse_args.input_dir):
        body_mask = None
        if not hold_out:
            create_segmentation_label_map(id_)
            body_mask = get_body_mask(id_)

        if norm_type == "z":
            z_normalise_input_img(id_, body_mask)
        elif norm_type == "global":
            range_normalize_input_img(id_, -1000, 29014) # Global min across all images is 1,000. Max is 29014 (due to artifacts)
        elif norm_type == "range":
            ## Normalise a between a range -> e.g -100,800 -> -1,1
            range_normalize_input_img(id_, min, max)
        elif norm_type == "range_0_mean":
            range_0_mean_input_img(id_, min, max)

    
    task_dir = ""
    if parse_args.task_id:
        task_dir = "/Task" + parse_args.task_id

    output_dataframe.index.name = 'id'
    os.makedirs('assets/Cervical_data' + task_dir, exist_ok=True)
    if not hold_out:
        train_index = output_dataframe.loc[train_ids]
        train_index.to_csv('assets/Cervical_data' + task_dir + '/data_index_train.csv')
        valid_index = output_dataframe.loc[valid_ids]
        valid_index.to_csv('assets/Cervical_data' + task_dir + '/data_index_valid.csv')
        test_index = output_dataframe.loc[test_ids]
        test_index.to_csv('assets/Cervical_data' + task_dir + '/data_index_test.csv')
    else:
        test_index = output_dataframe.loc[hold_out_ids]
        test_index.to_csv('assets/Cervical_data' + task_dir + '/data_index_test.csv')


