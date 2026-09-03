import cv2 
import tqdm
import torch
import numpy as np 
import matplotlib.pyplot as plt
import os 

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.semantic_refine_net import SemanticRefineNet
from models.losses import HybridLoss
from data.dataset import RefineDataset

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['image'].to(device)
        base_alpha = batch['base_alpha'].to(device)
        gt_alpha = batch['gt_alpha'].to(device)

        sem_label = batch.get('sem_label', None)
        if sem_label is not None:
            sem_label = sem_label.to(device)

        pred_alpha = model(images, base_alpha)

        loss = criterion(pred_alpha, gt_alpha, base_alpha, sem_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    total_transition_error = 0
    total_certain_error = 0
    total_pixels = 0
    total_transition_pixels = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            base_alpha = batch['base_alpha'].to(device)
            gt_alpha = batch['gt_alpha'].to(device)
            sem_label = batch.get('sem_label', None)
            if sem_label is not None:
                sem_label = sem_label.to(device)

            pred_alpha = model(images, base_alpha)
            loss = criterion(pred_alpha, gt_alpha, base_alpha, sem_label)
            total_loss += loss.item()

            pred_np = pred_alpha.cpu().numpy()
            gt_np = gt_alpha.cpu().numpy()

            transition_mask = (gt_np > 0.05) & (gt_np < 0.95)
            transition_error = np.abs(pred_np - gt_np)[transition_mask].mean() if transition_mask.any() else 0
            total_transition_error += transition_error * transition_mask.sum()
            total_transition_pixels += transition_mask.sum()

            certain_mask = (gt_np <= 0.05) | (gt_np >= 0.95)
            certain_error = np.abs(pred_np - gt_np)[certain_mask].mean() if certain_mask.any() else 0
            total_certain_error += certain_error * certain_mask.sum()
            total_pixels += certain_mask.sum()

    avg_transition_error = total_transition_error / total_transition_pixels if total_transition_pixels > 0 else 0
    avg_certain_error = total_certain_error / total_pixels if total_pixels > 0 else 0

    return {
        'loss': total_loss / len(dataloader),
        'transition_error': avg_transition_error,
        'certain_error': avg_certain_error
    }
    
def visualize_results(model, dataloader, device, num_samples=4, save_path=None):
    model.eval()

    samples = []
    for i, batch in enumerate(dataloader):
        if len(samples) >= num_samples:
            break
        sample = {
            'image': batch['image'][0:1],
            'base_alpha': batch['base_alpha'][0:1],
            'gt_alpha': batch['gt_alpha'][0:1],
            'name': batch['name'][0] if 'name' in batch else str(i)
        }
        samples.append(sample)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, sample in enumerate(samples):
            images = sample['image'].to(device)
            base_alpha = sample['base_alpha'].to(device)
            gt_alpha = sample['gt_alpha'].cpu().numpy().squeeze()

            pred_alpha = model(images, base_alpha)
            pred_alpha = pred_alpha.cpu().numpy().squeeze()
            base_alpha_np = base_alpha.cpu().numpy().squeeze()

            img_np = images[0].cpu().permute(1, 2, 0).numpy()

            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f'Original', fontsize=10)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(base_alpha_np, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title('Base Alpha (RVM)', fontsize=10)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_alpha, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title('Predicted Alpha', fontsize=10)
            axes[i, 2].axis('off')

            axes[i, 3].imshow(gt_alpha, cmap='gray', vmin=0, vmax=1)
            axes[i, 3].set_title('Ground Truth', fontsize=10)
            axes[i, 3].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization results to: {save_path}")
    
def compute_refinement_metrics(model, dataloader, device):
    model.eval()

    total_rvm_transition_error = 0
    total_our_transition_error = 0
    total_transition_pixels = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing metrics"):
            images = batch['image'].to(device)
            base_alpha = batch['base_alpha'].to(device)
            gt_alpha = batch['gt_alpha'].cpu().numpy().squeeze()

            pred_alpha = model(images, base_alpha)
            pred_alpha = pred_alpha.cpu().numpy().squeeze()
            base_alpha_np = base_alpha.cpu().numpy().squeeze()

            transition_mask = (gt_alpha > 0.05) & (gt_alpha < 0.95)

            if transition_mask.any():
                rvm_error = np.abs(base_alpha_np - gt_alpha)[transition_mask].sum()
                our_error = np.abs(pred_alpha - gt_alpha)[transition_mask].sum()

                total_rvm_transition_error += rvm_error
                total_our_transition_error += our_error
                total_transition_pixels += transition_mask.sum()

    avg_rvm_error = total_rvm_transition_error / total_transition_pixels
    avg_our_error = total_our_transition_error / total_transition_pixels
    improvement = (avg_rvm_error - avg_our_error) / avg_rvm_error * 100

    print(f"\nTransition Region Refinement Metrics:")
    print(f"   RVM average error: {avg_rvm_error:.4f}")
    print(f"   Our model error: {avg_our_error:.4f}")
    print(f"   Improvement: {improvement:.2f}%")

    return {
        'rvm_error': avg_rvm_error,
        'our_error': avg_our_error,
        'improvement': improvement
    }
    
def main():
    DATA_ROOT = "/content/drive/MyDrive/DATA/P3M-10k"
    SAVE_DIR = "/content/drive/MyDrive/RVM/semantic_refine_model"

    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    IMG_SIZE = (512, 512)

    TEST_MODE = True
    TEST_SAMPLES = 9421

    TRANSITION_WEIGHT = 3.0
    CERTAIN_WEIGHT = 0.2

    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nLoading training dataset...")
    train_dataset = RefineDataset(DATA_ROOT, split='train', size=IMG_SIZE, use_confidence_weight=True)

    if TEST_MODE:
        print(f"Test mode: using only first {TEST_SAMPLES} samples")
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(TEST_SAMPLES, len(train_dataset))))

    print("Loading validation dataset...")
    val_dataset = RefineDataset(DATA_ROOT, split='val', size=IMG_SIZE, use_confidence_weight=True)

    if TEST_MODE and len(val_dataset) > TEST_SAMPLES:
        val_dataset = Subset(val_dataset, range(TEST_SAMPLES))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, drop_last=True)

    print(f"\nTraining batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    print("\nBuilding Semantic Refine Network...")
    model = SemanticRefineNet().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    criterion = HybridLoss(
        transition_weight=TRANSITION_WEIGHT,
        certain_weight=CERTAIN_WEIGHT,
        grad_weight=0.1
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    print("\nStarting training!")
    print("=" * 60)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_transition_error': [], 'val_certain_error': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 40)

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        val_stats = validate(model, val_loader, criterion, device)

        scheduler.step(val_stats['loss'])

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_stats['loss'])
        history['val_transition_error'].append(val_stats['transition_error'])
        history['val_certain_error'].append(val_stats['certain_error'])

        print(f"\nEpoch {epoch} Results:")
        print(f"   Training loss: {train_loss:.4f}")
        print(f"   Validation loss: {val_stats['loss']:.4f}")
        print(f"   Transition error: {val_stats['transition_error']:.4f}")
        print(f"   Certain error: {val_stats['certain_error']:.4f}")
        print(f"   Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        if epoch % 4 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_stats['loss'],
                'train_loss': train_loss
            }, checkpoint_path)
            print(f"Model saved: {checkpoint_path}")

        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            best_path = os.path.join(SAVE_DIR, "model_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_stats['loss']
            }, best_path)
            print(f"Best model updated!")

        if epoch % 5 == 0:
            visualize_results(model, val_loader, device, num_samples=2,
                            save_path=os.path.join(SAVE_DIR, f"vis_epoch_{epoch}.png"))

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"All models saved in: {SAVE_DIR}")

    print("\nFinal evaluation on validation set:")
    metrics = compute_refinement_metrics(model, val_loader, device)

    import json
    with open(os.path.join(SAVE_DIR, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history, metrics


if __name__ == "__main__":
    model, history, metrics = main()