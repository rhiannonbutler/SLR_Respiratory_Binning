import os
import re
import glob
import subprocess
import numpy as np
import nibabel as nib

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Parent directory containing rep1, rep2, and rep3 folders
base_dir = "/home/rhiannon.b/Downloads/ReconData/sub9021/lumbar_siemens/niftis/tmp_dcm2bids/sub-sub9021/raw"

# Names of repetition folders
rep_folders = ["rep1", "rep2", "rep3"]

# Output directory for averaged files and masks
output_dir = os.path.join(base_dir, "rep_averaged")
os.makedirs(output_dir, exist_ok=True)


# ==============================================================================
# STEP 1: Match & Average Repetitions per Echo
# ==============================================================================
def get_echo_number(filename):
    """Extracts the echo index from filenames ending in _e<N>.nii.gz"""
    match = re.search(r'_e(\d+)\.nii\.gz$', filename)
    return int(match.group(1)) if match else None


def average_repetitions(base_dir, rep_folders, output_dir):
    print("--- Step 1: Averaging Repetitions per Echo ---")
    echo_groups = {}

    # Match files across reps by echo number
    for rep_idx, folder in enumerate(rep_folders):
        folder_path = os.path.join(base_dir, folder)
        for fpath in glob.glob(os.path.join(folder_path, "*.nii.gz")):
            echo_num = get_echo_number(os.path.basename(fpath))
            if echo_num is not None:
                if echo_num not in echo_groups:
                    echo_groups[echo_num] = [None] * len(rep_folders)
                echo_groups[echo_num][rep_idx] = fpath

    averaged_echo_vols = {}
    ref_img = None

    for echo_num, paths in sorted(echo_groups.items()):
        if any(p is None for p in paths):
            missing = [rep_folders[i] for i, p in enumerate(paths) if p is None]
            print(f"Skipping Echo {echo_num}: missing from {missing}")
            continue

        imgs = [nib.load(p) for p in paths]
        if ref_img is None:
            ref_img = imgs[0]

        # Voxel-wise mean across repetition volumes
        mean_3d_data = np.mean([img.get_fdata() for img in imgs], axis=0)
        
        # Save individual averaged echo file
        out_path = os.path.join(output_dir, f"avg_e{echo_num}.nii.gz")
        nib.save(nib.Nifti1Image(mean_3d_data, ref_img.affine, ref_img.header), out_path)
        print(f"Saved Averaged Echo {echo_num}: shape {mean_3d_data.shape} -> {out_path}")

        averaged_echo_vols[echo_num] = mean_3d_data

    return averaged_echo_vols, ref_img


# ==============================================================================
# STEP 2: Multi-Echo RMS Combination
# ==============================================================================
def compute_echo_rms(averaged_echo_vols, ref_img, output_dir):
    print("\n--- Step 2: Multi-Echo RMS Combination ---")
    echoes = sorted(list(averaged_echo_vols.keys()))

    if len(echoes) < 4:
        print(f"Error: Found only {len(echoes)} echoes. Need 4 for RMS combination.")
        return None, None

    # Stack 3D echo volumes into 4D array along the last dimension -> (X, Y, Z, 4)
    echo_stack_4d = np.stack([averaged_echo_vols[e] for e in range(1, 5)], axis=-1)
    print(f"4D Multi-Echo Volume Shape: {echo_stack_4d.shape} (X, Y, Z, Echo)")

    # Root Mean Square across echo axis (axis=-1)
    rms_4echo = np.sqrt(np.mean(np.square(echo_stack_4d), axis=-1))
    rms_3echo = np.sqrt(np.mean(np.square(echo_stack_4d[..., :3]), axis=-1))

    print(f"Final 4-Echo RMS Shape: {rms_4echo.shape}")
    print(f"Final 3-Echo RMS Shape: {rms_3echo.shape}")

    path_rms4 = os.path.join(output_dir, "rms_4echoes_3D.nii.gz")
    path_rms3 = os.path.join(output_dir, "rms_3echoes_3D.nii.gz")

    nib.save(nib.Nifti1Image(rms_4echo, ref_img.affine, ref_img.header), path_rms4)
    nib.save(nib.Nifti1Image(rms_3echo, ref_img.affine, ref_img.header), path_rms3)

    print(f"Saved 4-Echo RMS Volume: {path_rms4}")
    print(f"Saved 3-Echo RMS Volume: {path_rms3}")

    return path_rms4, path_rms3


# ==============================================================================
# STEP 3: Spinal Cord Toolbox (SCT) Segmentation
# ==============================================================================
def run_sct_segmentation(input_nii, task_type, output_suffix):
    out_mask = input_nii.replace(".nii.gz", f"_{output_suffix}.nii.gz")
    
    cmd_modern = ["sct_deepseg", "-i", input_nii, "-task", task_type, "-o", out_mask]
    legacy_bin = "sct_deepseg_sc" if "sc" in task_type or "spinal" in task_type else "sct_deepseg_gm"
    cmd_legacy = [legacy_bin, "-i", input_nii, "-o", out_mask]

    print(f"Running SCT command on {os.path.basename(input_nii)}...")
    try:
        subprocess.run(cmd_modern, capture_output=True, text=True, check=True)
        print(f"Success: Created {os.path.basename(out_mask)}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(cmd_legacy, capture_output=True, text=True, check=True)
            print(f"Success (Legacy SCT): Created {os.path.basename(out_mask)}")
        except Exception as e:
            print(f"SCT execution failed. Ensure SCT is added to your system PATH. Error: {e}")


# ==============================================================================
# MAIN EXECUTION FLOW
# ==============================================================================
if __name__ == "__main__":
    # Step 1: Average across repetitions
    echo_vols, reference_img = average_repetitions(base_dir, rep_folders, output_dir)

    # Step 2: Calculate RMS combinations across echoes
    path_rms4, path_rms3 = compute_echo_rms(echo_vols, reference_img, output_dir)

    # Step 3: Run SCT Segmentation if RMS files were generated
    if path_rms4 and os.path.exists(path_rms4):
        print("\n--- Step 3: Running SCT DeepSeg Mask Generation ---")
        run_sct_segmentation(path_rms4, task_type="seg_sc_mri", output_suffix="sc_mask")
        run_sct_segmentation(path_rms4, task_type="seg_gm_mri", output_suffix="gm_mask")