import kagglehub
import os
import shutil

# Create the data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download latest version
path = kagglehub.dataset_download("maimunulkjisan/rice-leaf-disease-dataset")

# Copy the downloaded dataset to the local data directory
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join('data', item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("Dataset downloaded and copied to data directory.")