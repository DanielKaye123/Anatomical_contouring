import os
import SimpleITK as sitk
import pandas as pd
import numpy as np


intensity_df = pd.DataFrame()
min_clip = -100
max_clip = 200

data_dir = "/vol/biomedic3/bglocker/radiotherapy/kat100/nifti"
i = 0
for patient_name in os.listdir(data_dir):
    patient_path = os.path.join(data_dir, patient_name)
    image_path = os.path.join(patient_path, "image.nii.gz")

    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayViewFromImage(image)

    image_array = np.where((image_array >= min_clip) & (image_array <= max_clip), image_array, 0)
    #image_array = np.clip(image_array, min_clip, max_clip)

    image_array = image_array - min_clip
  
    flattened_scan = image_array.ravel()
    # Append the intensity values to the overall list
    temp_df = pd.DataFrame({'Intensity': flattened_scan})
    intensity_df = pd.concat([intensity_df, temp_df], ignore_index=True)
    i += 1
    if (i > 10):
        break
    
print("Mean intensity of data", intensity_df['Intensity'].mean())