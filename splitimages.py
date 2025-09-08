import os
import random
import shutil

def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
    categories = os.listdir(source_dir)
    for category in categories:
        category_path = os.path.join(source_dir, category)
        if not os.path.isdir(category_path):
            continue
        
        images = os.listdir(category_path)
        random.shuffle(images)
        
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]
        
        train_category_dir = os.path.join(train_dir, category)
        val_category_dir = os.path.join(val_dir, category)
        
        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(val_category_dir, exist_ok=True)
        
        for img in train_images:
            src = os.path.join(category_path, img)
            dst = os.path.join(train_category_dir, img)
            shutil.copy(src, dst)
        
        for img in val_images:
            src = os.path.join(category_path, img)
            dst = os.path.join(val_category_dir, img)
            shutil.copy(src, dst)

# Paths
source_directory = 'dataset_all'   # Folder where all images are placed by category
train_directory = 'dataset/train'
val_directory = 'dataset/val'

# Split data
split_data(source_directory, train_directory, val_directory, split_ratio=0.8)

print("Data split into train and validation sets!")