import argparse
from dataset.dataloader import get_loader
import numpy as np
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train')
    parser.add_argument('--data_root_path', default='/home/aditya/Data/ISLES-2022')  # update this
    parser.add_argument('--data_txt_path', default='/home/aditya/Projects/CLIPSeg/CLIPStrokeSeg/dataset/dataset_list')
    parser.add_argument('--dataset_list', nargs='+', default=['isles'])
    parser.add_argument('--cache_dataset', action='store_true')  # optional
    parser.add_argument('--cache_rate', type=float, default=1.0)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)

    # Spatial and intensity transform settings
    parser.add_argument('--roi_x', type=int, default=128)
    parser.add_argument('--roi_y', type=int, default=128)
    parser.add_argument('--roi_z', type=int, default=64)
    parser.add_argument('--space_x', type=float, default=1.0)
    parser.add_argument('--space_y', type=float, default=1.0)
    parser.add_argument('--space_z', type=float, default=1.0)
    parser.add_argument('--a_min', type=float, default=0.0)
    parser.add_argument('--a_max', type=float, default=1000.0)
    parser.add_argument('--b_min', type=float, default=0.0)
    parser.add_argument('--b_max', type=float, default=1.0)
    parser.add_argument('--num_samples', type=int, default=2)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    loader, _ = get_loader(args)

    # Create output directory
    os.makedirs("tests/output_slices", exist_ok=True)

    for batch in loader:
        print("Image shape:", batch["image"].shape)   # Expect: [B, 3, D, H, W]
        print("Label shape:", batch["label"].shape)   # Expect: [B, D, H, W]
        print("Name:", batch["name"])

        # Get the first sample in the batch
        image = batch["image"][0].cpu().numpy()  # [C, D, H, W]
        label = batch["label"][0].cpu().numpy()  # [C, D, H, W]
        name = batch["name"][0]

        # Remove channel dimension if present
        if image.ndim == 4:
            image = image[0]  # [D, H, W]
        if label.ndim == 4:
            label = label[0]  # [D, H, W]

        mid_slice = image.shape[0] // 2

        # Save image slice
        plt.imsave(f"tests/output_slices/{name}_image_slice.png", image[mid_slice], cmap='gray')
        # Save label slice
        plt.imsave(f"tests/output_slices/{name}_label_slice.png", label[mid_slice], cmap='hot')

        print(f"Saved slices for {name} to tests/output_slices/")

        # Visualize the middle slice of the first image and label in the batch
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("MRI Image (mid slice)")
        plt.imshow(image[mid_slice], cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Mask (mid slice)")
        plt.imshow(label[mid_slice], cmap='hot', alpha=0.7)
        plt.axis('off')

        plt.show()
        break
