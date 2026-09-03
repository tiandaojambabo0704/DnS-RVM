import os
import cv2
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import glob
from PIL import Image

from models.rvm_wrapper import load_rvm_model
from data.preprocess import alpha_to_4class_semantic, predict_alpha_with_rvm, process_dataset

P3M_ROOT = "/content/drive/MyDrive/DATA/P3M-10k" 

RVM_WEIGHT_PATH = "/content/drive/MyDrive/RVM/rvm_resnet50.pth"

GENERATE_SEM_LABEL = False     
GENERATE_BASE_ALPHA = True    

TEST_MODE = True         
TEST_SIZE = 9421              
RGB_SAVE_LIMIT = 1          

ALPHA_BG_MAX = 0.05            
ALPHA_TR_BACK_MAX = 0.5        
ALPHA_TR_FORE_MIN = 0.5       
ALPHA_FORE_MIN = 0.95          

COLOR_MAP = {
    0: [0, 0, 0],          
    1: [255, 0, 0],        
    2: [0, 0, 255],         
    3: [255, 255, 255]      
}

print(f"Path: {P3M_ROOT}")
print(f"RVM Weight: {RVM_WEIGHT_PATH}")
print(f"Generate sem label: {GENERATE_SEM_LABEL}")
print(f"generate base alpha: {GENERATE_BASE_ALPHA}")
print(f"test mode: {TEST_MODE} ({TEST_SIZE})")
print(f"RGB save {RGB_SAVE_LIMIT} ")

def main():
    global GENERATE_BASE_ALPHA
    print("=" * 60)
    print(" P3M-10k data preprocess")
    print(f"   sem_label: {GENERATE_SEM_LABEL}")
    print(f"   base_alpha: {GENERATE_BASE_ALPHA}")
    print("=" * 60)

    model = None
    device = None
    if GENERATE_BASE_ALPHA:
        model, device = load_rvm_model(RVM_WEIGHT_PATH)
        if model is None:
            print("Fail")
            GENERATE_BASE_ALPHA = False

    datasets = []

    train_mask = os.path.join(P3M_ROOT, "train", "mask")
    train_img = os.path.join(P3M_ROOT, "train", "blurred_image")
    train_sem_rgb = os.path.join(P3M_ROOT, "train", "sem_rgb")
    train_sem_label = os.path.join(P3M_ROOT, "train", "sem_label")
    train_base_alpha = os.path.join(P3M_ROOT, "train", "base_alpha")

    if os.path.exists(train_mask):
        datasets.append(("Train", train_mask, train_img, train_sem_rgb, train_sem_label, train_base_alpha))
        print(f"Add Train")

    val_p_mask = os.path.join(P3M_ROOT, "validation", "P3M-500-P", "mask")
    val_p_img = os.path.join(P3M_ROOT, "validation", "P3M-500-P", "blurred_image")
    val_p_sem_rgb = os.path.join(P3M_ROOT, "validation", "P3M-500-P", "sem_rgb")
    val_p_sem_label = os.path.join(P3M_ROOT, "validation", "P3M-500-P", "sem_label")
    val_p_base_alpha = os.path.join(P3M_ROOT, "validation", "P3M-500-P", "base_alpha")

    if os.path.exists(val_p_mask):
        datasets.append(("Val-P3M-500-P", val_p_mask, val_p_img, val_p_sem_rgb, val_p_sem_label, val_p_base_alpha))
        print(f"Add Validation P3M-500-P")

    t_p_mask = os.path.join(P3M_ROOT, "validation", "P3M-500-NP", "mask")
    t_p_img = os.path.join(P3M_ROOT, "validation", "P3M-500-NP", "original_image")
    t_p_sem_rgb = os.path.join(P3M_ROOT, "validation", "P3M-500-NP", "sem_rgb")
    t_p_sem_label = os.path.join(P3M_ROOT, "validation", "P3M-500-NP", "sem_label")
    t_p_base_alpha = os.path.join(P3M_ROOT, "validation", "P3M-500-NP", "base_alpha")

    if os.path.exists(val_p_mask):
        datasets.append(("Val-P3M-500-NP", t_p_mask, t_p_img, t_p_sem_rgb, t_p_sem_label, t_p_base_alpha))
        print(f"Add Test P3M-500-NP")


    if not datasets:
        print("\nNot found")
        return

    total_files = 0
    total_sem = 0
    total_base = 0

    for name, mask_dir, img_dir, sem_rgb_dir, sem_label_dir, base_alpha_dir in datasets:
        print(f"\n {name}")
        print(f"   Mask: {mask_dir}")
        print(f"   Origin: {img_dir}")

        if GENERATE_SEM_LABEL:
            print(f"   Sem RGB out: {sem_rgb_dir} ({RGB_SAVE_LIMIT})")
            print(f"   Sem label out: {sem_label_dir}")
        if GENERATE_BASE_ALPHA:
            print(f"   base_alpha out: {base_alpha_dir}")

        n_files, n_sem, n_base = process_dataset(
            mask_dir, img_dir, sem_rgb_dir, sem_label_dir, base_alpha_dir,
            model, device, name, max_workers=4  
        )

        total_files += n_files
        total_sem += n_sem
        total_base += n_base
        
    print("\n" + "=" * 60)
    print("Complete")
    print(f"   # of files: {total_files}")
    if GENERATE_SEM_LABEL:
        print(f"   sem label success: {total_sem}")
    if GENERATE_BASE_ALPHA:
        print(f"   base_alpha success: {total_base}")
    print("=" * 60)
    
if __name__ == "__main__":
    main()