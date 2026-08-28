import os
import re
import glob
import numpy as np
import nibabel as nib

base_dir = "/home/rhiannon.b/Downloads/ReconData/sub9027/niftis/tmp_dcm2bids/sub-9027/normorraw"  # Directory containing rep1, rep2, rep3
rep_folders = ["rep1", "rep2", "rep3"]
output_dir = os.path.join(base_dir, "rep_averaged")
os.makedirs(output_dir, exist_ok=True)

def get_suffix_key(filename):
    # Matches starting from 'siemensdicoms' to the end of the filename
    match = re.search(r'(siemensdicoms_.*\.nii\.gz)$', filename)
    return match.group(1) if match else filename

def get_echo_instance_key(filename):
    match = re.search(r'(_e\d+_i\d+)\.nii\.gz$', filename)
    return match.group(1) if match else None

# 3. Map common file suffixes to their respective full paths in each repetition
file_groups = {}

for rep_idx, folder in enumerate(rep_folders):
    folder_path = os.path.join(base_dir, folder)
    nii_files = glob.glob(os.path.join(folder_path, "*.nii.gz"))
    
    for fpath in nii_files:
        key = get_echo_instance_key(os.path.basename(fpath))
        if key:
            if key not in file_groups:
                file_groups[key] = [None] * len(rep_folders)
            file_groups[key][rep_idx] = fpath

# 4. Average matching volumes
for key, paths in file_groups.items():
    # Skip if any rep folder is missing this specific scan
    if any(p is None for p in paths):
        missing = [rep_folders[i] for i, p in enumerate(paths) if p is None]
        print(f"Skipping {key}: Missing from {missing}")
        continue
    
    # Load and average voxel values
    imgs = [nib.load(p) for p in paths]
    data_arrays = [img.get_fdata() for img in imgs]
    mean_data = np.mean(data_arrays, axis=0)
    
    # Save averaged NIfTI volume
    ref_img = imgs[0]
    avg_img = nib.Nifti1Image(mean_data, ref_img.affine, ref_img.header)
    
    out_filename = f"avg{key}.nii.gz"  # Output like: avg_e1_i00001.nii.gz
    out_path = os.path.join(output_dir, out_filename)
    nib.save(avg_img, out_path)
    print(f"Saved: {out_filename}")




























































