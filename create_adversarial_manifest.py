import os
import glob
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def build_manifest(target_per_class=8000):
    random.seed(42)
    
    # 1. Collect all positive (scam) images
    all_positives = glob.glob("*_image.jpg")
    print(f"Total positive images found: {len(all_positives)}")
    
    # 2. Collect all negative (legitimate & adversarial) images
    adv_base = glob.glob("adversarial/*")
    adv_comp = glob.glob("adversarial_composites/*.jpg")
    scam_memes = glob.glob("assets/scam_memes/*.jpg")
    elon_x = glob.glob("assets/elon_x_screenshots/*.jpg")
    jimmy = glob.glob("assets/jimmy_faces/*.jpg")
    sys_neg = glob.glob("negatives/*.jpg")
    
    all_negatives = []
    # Always include test2.png
    if os.path.exists("test2.png"):
        all_negatives.append("test2.png")
        
    all_negatives.extend(adv_comp)    # 3500 composites
    all_negatives.extend(scam_memes)  # ~250 memes
    all_negatives.extend(elon_x)      # ~150 elon/x
    all_negatives.extend(jimmy)       # ~120 jimmy
    all_negatives.extend(sys_neg)     # ~3000 wallpapers/natural
    all_negatives.extend(adv_base)    # ~11000 discord/cats/screenshots
    
    # Remove duplicates and ensure file exists
    all_negatives = list(dict.fromkeys([f for f in all_negatives if os.path.isfile(f)]))
    print(f"Total negative images pool: {len(all_negatives)}")
    
    # Sample balanced set
    n = min(len(all_positives), len(all_negatives), target_per_class)
    selected_pos = random.sample(all_positives, n)
    
    # Prioritize test2.png and composites in negatives
    priority_neg = ["test2.png"] + adv_comp + scam_memes + elon_x + jimmy
    priority_neg = [p for p in priority_neg if p in all_negatives]
    remaining_neg = [p for p in all_negatives if p not in set(priority_neg)]
    
    needed = n - len(priority_neg)
    selected_neg = priority_neg + random.sample(remaining_neg, max(0, needed))
    selected_neg = selected_neg[:n]
    
    print(f"Sampled {len(selected_pos)} Positives and {len(selected_neg)} Negatives.")
    
    data = []
    for f in selected_pos:
        data.append({"filepath": f, "label": 1, "class_name": "mrbeast_scam"})
    for f in selected_neg:
        data.append({"filepath": f, "label": 0, "class_name": "legitimate_or_adversarial"})
        
    df = pd.DataFrame(data)
    
    # Stratified Split: 75% train, 15% val, 10% test
    # Ensure test2.png is in train so the model directly learns it
    is_test2 = df["filepath"] == "test2.png"
    test2_row = df[is_test2].copy()
    rest_df = df[~is_test2].copy()
    
    train_rest, temp_df = train_test_split(rest_df, test_size=0.25, random_state=42, stratify=rest_df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.40, random_state=42, stratify=temp_df["label"])
    
    train_df = pd.concat([test2_row, train_rest], ignore_index=True)
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    manifest_path = "dataset_manifest.csv"
    full_df.to_csv(manifest_path, index=False)
    
    print(f"\nSaved updated manifest to {manifest_path}:")
    print(f"  Train: {len(train_df)} (Pos: {(train_df['label']==1).sum()}, Neg: {(train_df['label']==0).sum()})")
    print(f"  Val:   {len(val_df)} (Pos: {(val_df['label']==1).sum()}, Neg: {(val_df['label']==0).sum()})")
    print(f"  Test:  {len(test_df)} (Pos: {(test_df['label']==1).sum()}, Neg: {(test_df['label']==0).sum()})")
    print(f"  Total: {len(full_df)}")

if __name__ == "__main__":
    build_manifest(target_per_class=8000)
