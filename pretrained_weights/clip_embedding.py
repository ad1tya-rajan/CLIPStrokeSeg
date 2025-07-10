import os
import clip
import torch
import pandas as pd


## PAOT
# ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus', 
#                 'Liver', 'Stomach', 'Arota', 'Postcava', 'Portal Vein and Splenic Vein',
#                 'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum', 'Hepatic Vessel',
#                 'Right Lung', 'Left Lung', 'Colon', 'Intestine', 'Rectum', 
#                 'Bladder', 'Prostate', 'Left Head of Femur', 'Right Head of Femur', 'Celiac Truck',
#                 'Kidney Tumor', 'Liver Tumor', 'Pancreas Tumor', 'Hepatic Vessel Tumor', 'Lung Tumor', 
#                 'Colon Tumor', 'Kidney Cyst']

# Read labels from TSV file
labels_df = pd.read_csv('/home/aditya/Projects/CLIPSeg/CLIPStrokeSeg/dataset/dataset_list/labels.tsv', sep='\t')

# Function to generate prompts for each row in the DataFrame
def generate_prompt(row):
    return f"A brain MRI scan showing a {row['side']} hemispheric lesion of volume: {row['volume (mm^3)']} mm^3."

prompts = [generate_prompt(row) for _, row in labels_df.iterrows()]
# print(prompts[0:5])  # Print the first 5 prompts for verification

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


text_inputs = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(device)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)        # Normalize the features

    print(text_features.shape, text_features.dtype)
    # print("Text features shape:", text_features.shape)
    # print("First feature vector:", text_features[0][:5])
    torch.save(text_features, 'mri_txt_encoding.pth')

