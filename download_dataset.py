from datasets import load_dataset

# Specify the folder where you want to store the dataset
store_folder = "/scratch/fem17004/sas20048/LLaVA_dataset"

# Load and download the dataset
dataset = load_dataset("liuhaotian/LLaVA-CC3M-Pretrain-595K", cache_dir=store_folder)

print(f"Dataset stored in: {store_folder}")
