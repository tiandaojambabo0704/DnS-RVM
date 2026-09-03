import numpy as np 
import pandas as pd 
import os 
import glob
import cv2
from image_utils import refine_alpha
import tqdm 
import torch
from skimage.metrics import mean_squared_error

def compute_metrics(pred, gt):
    pred = np.clip(pred, 0, 1)
    gt = np.clip(gt, 0, 1)
    mse = mean_squared_error(gt, pred)
    sad = np.sum(np.abs(pred - gt))
    return mse, sad

def evaluate_testset(testset_root, rvm_model, your_model, device, delta=0.7, target_size=(512, 512)):
    image_dir = os.path.join(testset_root, 'original_image')
    if not os.path.exists(image_dir):
        image_dir = os.path.join(testset_root, 'blurred_image')

    mask_dir = os.path.join(testset_root, 'mask')

    valid_pairs = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img_path in glob.glob(os.path.join(image_dir, ext)):
            name = os.path.basename(img_path).split('.')[0]
            mask_path = os.path.join(mask_dir, f"{name}.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_dir, f"{name}.jpg")
            if os.path.exists(mask_path):
                valid_pairs.append((img_path, mask_path, name))

    valid_pairs = sorted(valid_pairs)
    print(f"\n Find {len(valid_pairs)} images")
    print(f"   🔧 Refinement δ = {delta}")
    print("=" * 60)

    results = []

    for img_path, mask_path, name in tqdm(valid_pairs, desc="Evaluating"):
        image = cv2.imread(img_path)
        gt_alpha = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_alpha = gt_alpha.astype(np.float32) / 255.0

        if image is None or gt_alpha is None:
            continue

        rvm_alpha = rvm_model.predict(image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, target_size)
        base_resized = cv2.resize(rvm_alpha, target_size)

        img_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        base_tensor = torch.from_numpy(base_resized).unsqueeze(0).unsqueeze(0).float()

        img_tensor = img_tensor.to(device)
        base_tensor = base_tensor.to(device)

        with torch.no_grad():
            semantic_alpha = your_model(img_tensor, base_tensor)
            semantic_alpha = semantic_alpha.squeeze().cpu().numpy()

        h, w = image.shape[:2]
        semantic_alpha = cv2.resize(semantic_alpha, (w, h))
        semantic_alpha = np.clip(semantic_alpha, 0, 1)

        refined_alpha = refine_alpha(rvm_alpha, semantic_alpha, delta=delta)

        rvm_mse, rvm_sad = compute_metrics(rvm_alpha, gt_alpha)
        refined_mse, refined_sad = compute_metrics(refined_alpha, gt_alpha)

        transition_mask = (gt_alpha > 0.05) & (gt_alpha < 0.95)
        if transition_mask.any():
            rvm_trans_mse = mean_squared_error(gt_alpha[transition_mask], rvm_alpha[transition_mask])
            rvm_trans_sad = np.sum(np.abs(rvm_alpha[transition_mask] - gt_alpha[transition_mask]))
            refined_trans_mse = mean_squared_error(gt_alpha[transition_mask], refined_alpha[transition_mask])
            refined_trans_sad = np.sum(np.abs(refined_alpha[transition_mask] - gt_alpha[transition_mask]))
        else:
            rvm_trans_mse = rvm_trans_sad = refined_trans_mse = refined_trans_sad = 0

        results.append({
            'name': name,
            'rvm_mse': rvm_mse,
            'rvm_sad': rvm_sad,
            'rvm_trans_mse': rvm_trans_mse,
            'rvm_trans_sad': rvm_trans_sad,
            'refined_mse': refined_mse,
            'refined_sad': refined_sad,
            'refined_trans_mse': refined_trans_mse,
            'refined_trans_sad': refined_trans_sad,
            'improvement_mse': rvm_mse - refined_mse,
            'improvement_sad': rvm_sad - refined_sad,
            'improvement_percent_mse': ((rvm_mse - refined_mse) / rvm_mse * 100) if rvm_mse > 0 else 0,
            'improvement_percent_sad': ((rvm_sad - refined_sad) / rvm_sad * 100) if rvm_sad > 0 else 0,
        })

    df = pd.DataFrame(results)
    summary = {
        'total_images': len(df),
        'rvm_avg_mse': df['rvm_mse'].mean(),
        'rvm_avg_sad': df['rvm_sad'].mean(),
        'rvm_avg_trans_mse': df['rvm_trans_mse'].mean(),
        'rvm_avg_trans_sad': df['rvm_trans_sad'].mean(),
        'refined_avg_mse': df['refined_mse'].mean(),
        'refined_avg_sad': df['refined_sad'].mean(),
        'refined_avg_trans_mse': df['refined_trans_mse'].mean(),
        'refined_avg_trans_sad': df['refined_trans_sad'].mean(),
        'avg_improvement_mse': df['improvement_mse'].mean(),
        'avg_improvement_sad': df['improvement_sad'].mean(),
        'avg_improvement_percent_mse': df['improvement_percent_mse'].mean(),
        'avg_improvement_percent_sad': df['improvement_percent_sad'].mean(),
        'improved_count_mse': (df['improvement_mse'] > 0).sum(),
        'improved_count_sad': (df['improvement_sad'] > 0).sum(),
    }

    return df, summary

def evaluate_testset_with_precomputed(testset_root, your_model, device,
                                       delta=0.7, target_size=(512, 512),
                                       base_alpha_dir_name="base_alpha"):
    
    image_dir = os.path.join(testset_root, 'original_image')
    if not os.path.exists(image_dir):
        image_dir = os.path.join(testset_root, 'blurred_image')

    mask_dir = os.path.join(testset_root, 'mask')
    base_alpha_dir = os.path.join(testset_root, base_alpha_dir_name)

    if not os.path.exists(base_alpha_dir):
        return evaluate_testset(testset_root, None, your_model, device, delta, target_size)

    valid_pairs = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img_path in glob.glob(os.path.join(image_dir, ext)):
            name = os.path.basename(img_path).split('.')[0]
            mask_path = os.path.join(mask_dir, f"{name}.png")
            base_path = os.path.join(base_alpha_dir, f"{name}.png")

            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_dir, f"{name}.jpg")
            if not os.path.exists(base_path):
                base_path = os.path.join(base_alpha_dir, f"{name}.jpg")

            if os.path.exists(mask_path) and os.path.exists(base_path):
                valid_pairs.append((img_path, mask_path, base_path, name))

    valid_pairs = sorted(valid_pairs)
    print(f"\n Find {len(valid_pairs)} images")
    print(f"   pre-generate base_alpha: {base_alpha_dir}")
    print(f"   🔧 Refinement δ = {delta}")
    print("=" * 60)

    results = []

    for img_path, mask_path, base_path, name in tqdm(valid_pairs, desc="Evaluating"):
        image = cv2.imread(img_path)
        gt_alpha = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_alpha = gt_alpha.astype(np.float32) / 255.0

        base_alpha = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
        base_alpha = base_alpha.astype(np.float32) / 255.0

        if image is None or gt_alpha is None or base_alpha is None:
            print(f"   Skip {name}")
            continue

        h, w = image.shape[:2]
        if base_alpha.shape[:2] != (h, w):
            base_alpha = cv2.resize(base_alpha, (w, h))

        rvm_alpha = base_alpha

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, target_size)
        base_resized = cv2.resize(rvm_alpha, target_size)

        img_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        base_tensor = torch.from_numpy(base_resized).unsqueeze(0).unsqueeze(0).float()

        img_tensor = img_tensor.to(device)
        base_tensor = base_tensor.to(device)

        with torch.no_grad():
            semantic_alpha = your_model(img_tensor, base_tensor)
            semantic_alpha = semantic_alpha.squeeze().cpu().numpy()
            semantic_alpha = cv2.resize(semantic_alpha, (w, h))
            semantic_alpha = np.clip(semantic_alpha, 0, 1)

        refined_alpha = refine_alpha(rvm_alpha, semantic_alpha, delta=delta)

        rvm_mse, rvm_sad = compute_metrics(rvm_alpha, gt_alpha)
        refined_mse, refined_sad = compute_metrics(refined_alpha, gt_alpha)

        transition_mask = (gt_alpha > 0.05) & (gt_alpha < 0.95)
        if transition_mask.any():
            rvm_trans_mse = mean_squared_error(gt_alpha[transition_mask], rvm_alpha[transition_mask])
            rvm_trans_sad = np.sum(np.abs(rvm_alpha[transition_mask] - gt_alpha[transition_mask]))
            refined_trans_mse = mean_squared_error(gt_alpha[transition_mask], refined_alpha[transition_mask])
            refined_trans_sad = np.sum(np.abs(refined_alpha[transition_mask] - gt_alpha[transition_mask]))
        else:
            rvm_trans_mse = rvm_trans_sad = refined_trans_mse = refined_trans_sad = 0

        results.append({
            'name': name,
            'rvm_mse': rvm_mse,
            'rvm_sad': rvm_sad,
            'rvm_trans_mse': rvm_trans_mse,
            'rvm_trans_sad': rvm_trans_sad,
            'refined_mse': refined_mse,
            'refined_sad': refined_sad,
            'refined_trans_mse': refined_trans_mse,
            'refined_trans_sad': refined_trans_sad,
            'improvement_mse': rvm_mse - refined_mse,
            'improvement_sad': rvm_sad - refined_sad,
            'improvement_percent_mse': ((rvm_mse - refined_mse) / rvm_mse * 100) if rvm_mse > 0 else 0,
            'improvement_percent_sad': ((rvm_sad - refined_sad) / rvm_sad * 100) if rvm_sad > 0 else 0,
        })

    df = pd.DataFrame(results)
    summary = {
        'total_images': len(df),
        'rvm_avg_mse': df['rvm_mse'].mean(),
        'rvm_avg_sad': df['rvm_sad'].mean(),
        'rvm_avg_trans_mse': df['rvm_trans_mse'].mean(),
        'rvm_avg_trans_sad': df['rvm_trans_sad'].mean(),
        'refined_avg_mse': df['refined_mse'].mean(),
        'refined_avg_sad': df['refined_sad'].mean(),
        'refined_avg_trans_mse': df['refined_trans_mse'].mean(),
        'refined_avg_trans_sad': df['refined_trans_sad'].mean(),
        'avg_improvement_mse': df['improvement_mse'].mean(),
        'avg_improvement_sad': df['improvement_sad'].mean(),
        'avg_improvement_percent_mse': df['improvement_percent_mse'].mean(),
        'avg_improvement_percent_sad': df['improvement_percent_sad'].mean(),
        'improved_count_mse': (df['improvement_mse'] > 0).sum(),
        'improved_count_sad': (df['improvement_sad'] > 0).sum(),
    }

    return df, summary

def print_summary(summary):
    print("\n" + "=" * 60)
    print("Evaluation summary")
    print("=" * 60)
    print(f"\nTotal images: {summary['total_images']}")

    print("\n" + "-" * 40)
    print("MSE (Mean Squared Error)")
    print("-" * 40)
    print(f"   RVM:     {summary['rvm_avg_mse']:.6f}")
    print(f"   Semantic Refined: {summary['refined_avg_mse']:.6f}")
    print(f"   Improvement:           {summary['avg_improvement_mse']:.6f} ({summary['avg_improvement_percent_mse']:.2f}%)")
    print(f"   Improved count:     {summary['improved_count_mse']}/{summary['total_images']}")

    print("\n" + "-" * 40)
    print("SAD (Sum of Absolute Differences)")
    print("-" * 40)
    print(f"   RVM:     {summary['rvm_avg_sad']:.2f}")
    print(f"   Semantic Refined: {summary['refined_avg_sad']:.2f}")
    print(f"   Improvement:           {summary['avg_improvement_sad']:.2f} ({summary['avg_improvement_percent_sad']:.2f}%)")
    print(f"   Improved count:     {summary['improved_count_sad']}/{summary['total_images']}")

    print("\n" + "-" * 40)
    print("Transition region (0.05 < α < 0.95)")
    print("-" * 40)
    print(f"   RVM MSE:     {summary['rvm_avg_trans_mse']:.6f}")
    print(f"   Semantic Refined MSE: {summary['refined_avg_trans_mse']:.6f}")
    print(f"   RVM SAD:     {summary['rvm_avg_trans_sad']:.2f}")
    print(f"   Semantic Refined SAD: {summary['refined_avg_trans_sad']:.2f}")
    print("=" * 60)
    
def plot_evaluation_results(df, summary, save_path=None):
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'DejaVu Sans'

    if df is None or len(df) == 0:
        print("No plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.hist(df['rvm_mse'], bins=30, alpha=0.5, label=f'RVM (mean={summary["rvm_avg_mse"]:.5f})', color='blue')
    ax1.hist(df['refined_mse'], bins=30, alpha=0.5, label=f'Refined (mean={summary["refined_avg_mse"]:.5f})', color='red')
    ax1.set_xlabel('MSE')
    ax1.set_ylabel('Frequency')
    ax1.set_title('MSE Distribution Comparison')
    ax1.legend()

    ax2 = axes[0, 1]
    ax2.hist(df['rvm_sad'], bins=30, alpha=0.5, label=f'RVM (mean={summary["rvm_avg_sad"]:.1f})', color='blue')
    ax2.hist(df['refined_sad'], bins=30, alpha=0.5, label=f'Refined (mean={summary["refined_avg_sad"]:.1f})', color='red')
    ax2.set_xlabel('SAD')
    ax2.set_ylabel('Frequency')
    ax2.set_title('SAD Distribution Comparison')
    ax2.legend()

    ax3 = axes[1, 0]
    improvements = df['improvement_mse']
    ax3.scatter(range(len(improvements)), improvements, alpha=0.5, s=10)
    ax3.axhline(y=0, color='red', linestyle='--', label='No Improvement')
    ax3.set_xlabel('Image Index')
    ax3.set_ylabel('MSE Improvement')
    ax3.set_title('MSE Improvement per Image')
    ax3.legend()

    ax4 = axes[1, 1]
    improved_count = (df['improvement_mse'] > 0).sum()
    worsened_count = (df['improvement_mse'] < 0).sum()
    no_change_count = (df['improvement_mse'] == 0).sum()

    bars = ax4.bar(['Improved', 'Worsened', 'No Change'],
                   [improved_count, worsened_count, no_change_count],
                   color=['green', 'red', 'gray'])
    ax4.set_ylabel('Number of Images')
    ax4.set_title(f'Improvement Statistics: {improved_count}/{len(df)} images improved')

    for bar, v in zip(bars, [improved_count, worsened_count, no_change_count]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 str(v), ha='center', fontweight='bold', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()