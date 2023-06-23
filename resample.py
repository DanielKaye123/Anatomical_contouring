import os
import argparse
import nibabel as nib
from scipy.ndimage import zoom
import glob
import numpy as np
import SimpleITK as sitk


def upsample(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        

    # Referance images to get correct upscale dimensions
    ref_dir = "/vol/biomedic3/bglocker/radiotherapy/kat100/nifti"

    #ref_dir = "/vol/biomedic3/bglocker/radiotherapy/kat10"
    
    non_prediciton = ('cov_diag.nii.gz', 'cov_factor.nii.gz', 'logit_mean.nii.gz', 'prob_maps.nii.gz', '.csv', 'random_sampler', 'cm.pickle', 'random_sampler.pickle', "dataset.json", "predict_from_raw_data_args.json", "plans.json", ".npz", ".pkl","upsampled", "patients")
    for prediciton in os.listdir(input_dir):
        if not prediciton.endswith(non_prediciton):
           # load the input image
           input_image = sitk.ReadImage(os.path.join(input_dir, prediciton))

           # Extract patient ID name from string. Get the input CT scan as a reference image
           patient_name = prediciton.split("_")[0] 
           ref_image = sitk.ReadImage(os.path.join(ref_dir, patient_name, "image.nii.gz"))

        #    patient_name = prediciton.split(".")[0] 
        #    ref_image = sitk.ReadImage(os.path.join(ref_dir, patient_name, patient_name + "_image.nii.gz"))
           
           # get the original spacing and orientation matrix
           original_spacing = ref_image.GetSpacing()
           original_orientation = ref_image.GetDirection()

    
           # upsample the downsampled image back to the original resolution
           upsampled_image= sitk.Resample(input_image, ref_image.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor, input_image.GetOrigin(),original_spacing, input_image.GetDirection(),0, input_image.GetPixelIDValue())
           upsampled_uint8 = sitk.Cast(upsampled_image, sitk.sitkUInt8)

            # Set the origin, spacing, and orientation of the upsampled image to match the reference image
           upsampled_uint8.SetOrigin(ref_image.GetOrigin())
           upsampled_uint8.SetSpacing(ref_image.GetSpacing())
           upsampled_uint8.SetDirection(ref_image.GetDirection())
    

           # save the upsampled image with its original orientation matrix and spacing
           sitk.WriteImage(upsampled_uint8, os.path.join(output_dir, prediciton))


def downsample(input_dir, output_dir, new_spacing = [2.0, 2.0, 2.0]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for sample in os.listdir(input_dir):
        sample_dir = os.path.join(input_dir, sample)
        print(sample_dir)
        for img in os.listdir(sample_dir):
            
            input_image = sitk.ReadImage((sample_dir) + "/" + img)
            
            # get the original image's spacing and size
            original_spacing = input_image.GetSpacing()
            original_size = input_image.GetSize()

            # compute the new size based on a 2mm isotropic resolution
            new_size = [int(round(original_size[i] * original_spacing[i] / 2)) for i in range(3)]
    

            # compute the new origin
            original_origin = input_image.GetOrigin()
            new_origin = [original_origin[i] + 0.5 * original_spacing[i] * (1 - float(new_size[i])/original_size[i]) for i in range(3)]

            # downsample the input volume
            if (img == "image.nii.gz"):
                resampled_image = sitk.Resample(input_image, new_size, sitk.Transform(), sitk.sitkLinear, new_origin, new_spacing, input_image.GetDirection())
            else:
                # downsample the input mask using nearest-neighbor interpolation
                resampled_image = sitk.Resample(input_image, new_size, sitk.Transform(), sitk.sitkNearestNeighbor, new_origin, new_spacing, input_image.GetDirection())
            
            # Make output dir if it doesn't exist
            os.makedirs(os.path.join(output_dir, sample), exist_ok=True)

            # Save image
            sitk.WriteImage(resampled_image, os.path.join(output_dir, sample, img))
            


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
    parser.add_argument('--zoom-type',
                        required=True,
                        type=str,
                        help='u/d values to choose between upsampling or downsampling')
    
    parse_args, unknown = parser.parse_known_args()


    # Defaults to downsample
    if(parse_args.zoom_type.upper() == "U"):
        # Upsample scale factor
        upsample(parse_args.input_dir, parse_args.output_dir)
    else:
        # Downsample resolution [2.0,2.0,2.0]
        downsample(parse_args.input_dir, parse_args.output_dir, new_spacing = [2.0, 2.0, 2.0])

    
    
 

   