import os
import random
from glob import glob

root = "/home/aditya/Data/ISLES-2022"
output_dir = "/home/aditya/Projects/CLIPSeg/CLIPStrokeSeg/dataset/dataset_list"
os.makedirs(output_dir, exist_ok=True)

# Get all subject directories
subjects = sorted(glob(os.path.join(root, "sub-strokecase*")))
random.seed(42)  # for reproducibility
random.shuffle(subjects)

# Split ratios
n_total = len(subjects)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

train_subjects = subjects[:n_train]
val_subjects = subjects[n_train:n_train + n_val]
test_subjects = subjects[n_train + n_val:]

print(f"Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}")

def make_lines(subject_paths):
    lines = []
    for subject_path in subject_paths:
        sub_id = os.path.basename(subject_path)
        ses_dir = os.path.join(sub_id, "ses-0001")

        flair = os.path.join(ses_dir, "anat", f"{sub_id}_ses-0001_FLAIR.nii.gz")
        adc = os.path.join(ses_dir, "dwi", f"{sub_id}_ses-0001_adc.nii.gz")
        dwi = os.path.join(ses_dir, "dwi", f"{sub_id}_ses-0001_dwi.nii.gz")
        label = os.path.join("derivatives", sub_id, "ses-0001", f"{sub_id}_ses-0001_msk.nii.gz")

        line = f"{flair},{adc},{dwi} {label}"
        lines.append(line)
    return lines

splits = {
    'train': make_lines(train_subjects),
    'val': make_lines(val_subjects),
    'test': make_lines(test_subjects)
}

for split, lines in splits.items():
    path = os.path.join(output_dir, f"isles_{split}.txt")
    with open(path, "w") as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Wrote {len(lines)} lines to {path}")
