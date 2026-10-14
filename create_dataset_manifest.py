import os
import glob
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def build_manifest():
    random.seed(42)
    
    # 1. Gather all positive images (the original dataset images)
    all_positive_files = [
        f for f in glob.glob("*_image.jpg")
        if os.path.isfile(f)
    ]
    print(f"Total positive images available: {len(all_positive_files)}")
    
    # 2. Gather all negative images
    negative_files = glob.glob("negatives/*.jpg")
    print(f"Total negative images available: {len(negative_files)}")
    
    # Subsample positives to match negatives for a balanced binary dataset
    n_samples = len(negative_files)
    selected_positives = random.sample(all_positive_files, n_samples)
    
    data = []
    for f in selected_positives:
        data.append({
            "filepath": f,
            "label": 1,
            "class_name": "mrbeast_scam"
        })
        
    for f in negative_files:
        data.append({
            "filepath": f,
            "label": 0,
            "class_name": "legitimate_or_other"
        })
        
    df = pd.DataFrame(data)
    
    # 3. Stratified Train / Val / Test split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    )
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    manifest_path = "dataset_manifest.csv"
    full_df.to_csv(manifest_path, index=False)
    
    print(f"\nManifest saved to {manifest_path}:")
    print(f"  Train samples: {len(train_df)} (Pos: {(train_df['label']==1).sum()}, Neg: {(train_df['label']==0).sum()})")
    print(f"  Val samples:   {len(val_df)} (Pos: {(val_df['label']==1).sum()}, Neg: {(val_df['label']==0).sum()})")
    print(f"  Test samples:  {len(test_df)} (Pos: {(test_df['label']==1).sum()}, Neg: {(test_df['label']==0).sum()})")
    print(f"  Total samples: {len(full_df)}")

if __name__ == "__main__":
    build_manifest()
