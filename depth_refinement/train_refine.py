import os
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from refine_net import RefineNet
from rvm_with_refine import RVMWithRefine

from model.model import MattingNetwork


class P3M10KDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, image_size=512):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size

        if split == "train":
            self.image_dir = self.root_dir  / "train" / "blurred_image"
            self.mask_dir = self.root_dir  / "train" / "mask"
            self.list_path = self.root_dir / "train_list.txt"
        else:
            raise ValueError("training only")
        
        print(f"Loading dataset from {self.image_dir} and {self.mask_dir} with list {self.list_path}")
        
        with open(self.list_path, "r", encoding='utf-8') as f:
            self.names = [line.strip() for line in f if line.strip()]

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.names)
    
    def _find_image_path(self, folder, stem):
        """
        stem may not have the same name as the image file, so we need to find the image file by stem
        """
        candidates = [
            folder / f"{stem}.jpg",
            folder / f"{stem}.png",
            folder / f"{stem}.jpeg",
            folder / stem,
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"\nCannot find image for stem {stem} in {folder}\n")
    
    def __getitem__(self, idx):
        stem = self.names[idx]

        image_path = self._find_image_path(self.image_dir, stem)
        mask_path = self._find_image_path(self.mask_dir, stem)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) # [0, 255]

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        image = self.to_tensor(image) # [3, H, W] [0, 1]
        mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0  # [1, H, W] [0, 1]    

        return image, mask
    
def create_dataset(root_dir="P3M-10k", split="train", image_size=512):
    dataset = P3M10KDataset(root_dir=root_dir, split=split, image_size=image_size)
    print("dataset size =", len(dataset))

    image, mask = dataset[0]
    print("image shape:", image.shape)
    print("mask shape:", mask.shape)
    print("image min/max:", image.min().item(), image.max().item())
    print("mask min/max:", mask.min().item(), mask.max().item())

    return dataset

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    model.rvm.eval() # ensure rvm does not enter training mode
    total_loss = 0.0

    for images, alpha_gt in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        alpha_gt = alpha_gt.to(device)

        outputs = model(images, downsample_ratio=0.25)
        pha_refined = outputs['pha_refined']

        loss = criterion(pha_refined, alpha_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def smoke_test(model, dataloader, criterion, device):
    model.train()
    model.rvm.eval()

    images, alpha_gt = next(iter(dataloader))
    images = images.to(device)
    alpha_gt = alpha_gt.to(device)

    print("\n[Smoke Test]")
    print("images shape:", images.shape)
    print("alpha_gt shape:", alpha_gt.shape)

    outputs = model(images, downsample_ratio=0.25)

    pha = outputs["pha"]
    pha_refined = outputs["pha_refined"]
    boundary_mask = outputs["boundary_mask"]
    delta = outputs["delta"]

    print("pha shape:", pha.shape)
    print("pha_refined shape:", pha_refined.shape)
    print("boundary_mask shape:", boundary_mask.shape)
    print("delta shape:", delta.shape)

    print("pha min/max:", pha.min().item(), pha.max().item())
    print("pha_refined min/max:", pha_refined.min().item(), pha_refined.max().item())
    print("boundary_mask unique:", boundary_mask.unique()[:10])

    loss = criterion(pha_refined, alpha_gt)
    print("loss:", loss.item())

    loss.backward()
    print("backward success!\n")

def main():

    # Create dataset and dataloader
    train_dataset = create_dataset(root_dir="P3M-10k", split="train", image_size=512)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    checkpoint_path = Path("checkpoints") / "rvm_mobilenetv3.pth"

    rvm = MattingNetwork("mobilenetv3").to(device)
    rvm.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # freeze rvm parameters
    rvm.eval()
    for param in rvm.parameters():
        param.requires_grad = False

    # use our refine net
    refine_net = RefineNet(in_channels=4,hidden_channels=32).to(device)
    model = RVMWithRefine(rvm_model=rvm, refine_net=refine_net).to(device)
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(refine_net.parameters(), lr=1e-4)

    # run smoke test before training
    # smoke_test(model, train_loader, criterion, device)

    # start training
    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.10f}")

        save_path = f"checkpoints/refine_net_epoch_{epoch+1}.pth"
        torch.save(refine_net.state_dict(), save_path)
        print(f"Saved refine net checkpoint to {save_path}\n")


if __name__ == "__main__":
    main()