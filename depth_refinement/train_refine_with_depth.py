# import os
# from pathlib import Path

# import cv2
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from tqdm import tqdm

# from refine_net import RefineNet
# from rvm_with_refine import RVMWithRefine
# from model.model import MattingNetwork


# class P3M10KDepthDataset(Dataset):
#     def __init__(self, root_dir, split="train", image_size=512):
#         self.root_dir = Path(root_dir)
#         self.split = split
#         self.image_size = image_size

#         if split == "train":
#             self.image_dir = self.root_dir / "train" / "blurred_image"
#             self.mask_dir = self.root_dir / "train" / "mask"
#             self.depth_dir = self.root_dir / "train" / "depth"
#             self.list_path = self.root_dir / "train_list.txt"
#         else:
#             raise ValueError("training only for now")

#         print(f"Loading dataset:")
#         print(f"  image_dir = {self.image_dir}")
#         print(f"  mask_dir  = {self.mask_dir}")
#         print(f"  depth_dir = {self.depth_dir}")
#         print(f"  list_path = {self.list_path}")
#         print("image exists:", self.image_dir.exists())
#         print("mask exists :", self.mask_dir.exists())
#         print("depth exists:", self.depth_dir.exists())
#         print("list exists :", self.list_path.exists())

#         with open(self.list_path, "r", encoding="utf-8") as f:
#             self.names = [line.strip() for line in f if line.strip()]

#         self.to_tensor = transforms.ToTensor()

#     def __len__(self):
#         return len(self.names)

#     def _find_file(self, folder, stem):
#         candidates = [
#             folder / f"{stem}.png",
#             folder / f"{stem}.jpg",
#             folder / f"{stem}.jpeg",
#             folder / stem,
#         ]
#         for p in candidates:
#             if p.exists():
#                 return p
#         raise FileNotFoundError(f"Cannot find file for stem {stem} in {folder}")


#     def __getitem__(self, idx):
#         stem = self.names[idx]

#         image_path = self._find_file(self.image_dir, stem)
#         mask_path = self._find_file(self.mask_dir, stem)
#         depth_path = self._find_file(self.depth_dir, stem)

#         image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
#         if image is None:
#             raise ValueError(f"Cannot read image: {image_path}")
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             raise ValueError(f"Cannot read mask: {mask_path}")

#         depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
#         if depth is None:
#             raise ValueError(f"Cannot read depth: {depth_path}")

#         image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
#         mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
#         depth = cv2.resize(depth, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

#         image = self.to_tensor(image)  # [3,H,W], [0,1]
#         mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0   # [1,H,W]
#         depth = torch.from_numpy(depth).float().unsqueeze(0) / 255.0 # [1,H,W]

#         return image, mask, depth


# def create_dataset(root_dir="P3M-10k", split="train", image_size=512):
#     dataset = P3M10KDepthDataset(root_dir=root_dir, split=split, image_size=image_size)
#     print("dataset size =", len(dataset))

#     image, mask, depth = dataset[0]
#     print("image shape:", image.shape)
#     print("mask shape :", mask.shape)
#     print("depth shape:", depth.shape)
#     print("image min/max:", image.min().item(), image.max().item())
#     print("mask  min/max:", mask.min().item(), mask.max().item())
#     print("depth min/max:", depth.min().item(), depth.max().item())

#     return dataset


# def smoke_test(model, dataloader, criterion, device):
#     model.train()
#     model.rvm.eval()

#     images, alpha_gt, depth = next(iter(dataloader))
#     images = images.to(device, non_blocking=True)
#     alpha_gt = alpha_gt.to(device, non_blocking=True)
#     depth = depth.to(device, non_blocking=True)

#     print("\n[Smoke Test with Depth]")
#     print("images shape:", images.shape)
#     print("alpha_gt shape:", alpha_gt.shape)
#     print("depth shape:", depth.shape)

#     outputs = model(images, depth, downsample_ratio=0.25)

#     pha = outputs["pha"]
#     pha_refined = outputs["pha_refined"]
#     boundary_mask = outputs["boundary_mask"]
#     delta = outputs["delta"]

#     print("pha shape:", pha.shape)
#     print("pha_refined shape:", pha_refined.shape)
#     print("boundary_mask shape:", boundary_mask.shape)
#     print("delta shape:", delta.shape)

#     loss = criterion(pha_refined, alpha_gt)
#     print("loss:", loss.item())

#     loss.backward()
#     print("backward success!\n")


# def train_one_epoch(model, dataloader, optimizer, criterion, device):
#     model.train()
#     model.rvm.eval()

#     total_loss = 0.0
#     total_coarse_loss = 0.0
#     total_refined_loss = 0.0

#     for images, alpha_gt, depth in tqdm(dataloader, desc="Training"):
#         images = images.to(device, non_blocking=True)
#         alpha_gt = alpha_gt.to(device, non_blocking=True)
#         depth = depth.to(device, non_blocking=True)

#         outputs = model(images, depth, downsample_ratio=0.25)

#         pha = outputs["pha"]
#         pha_refined = outputs["pha_refined"]

#         coarse_loss = criterion(pha, alpha_gt)
#         refined_loss = criterion(pha_refined, alpha_gt)

#         loss = refined_loss

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         total_coarse_loss += coarse_loss.item()
#         total_refined_loss += refined_loss.item()

#     avg_loss = total_loss / len(dataloader)
#     avg_coarse = total_coarse_loss / len(dataloader)
#     avg_refined = total_refined_loss / len(dataloader)

#     return avg_loss, avg_coarse, avg_refined


# def main():
#     os.makedirs("checkpoints", exist_ok=True)

#     train_dataset = create_dataset(root_dir="P3M-10k", split="train", image_size=512)
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=4,
#         shuffle=True,
#         num_workers=0,
#         pin_memory=True,
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("\nUsing device:", device)

#     rvm_checkpoint_path = "checkpoints/rvm_mobilenetv3.pth"

#     rvm = MattingNetwork("mobilenetv3").to(device)
#     rvm.load_state_dict(torch.load(rvm_checkpoint_path, map_location=device))

#     rvm.eval()
#     for param in rvm.parameters():
#         param.requires_grad = False

#     refine_net = RefineNet(in_channels=5, hidden_channels=32).to(device)
#     model = RVMWithRefine(rvm_model=rvm, refine_net=refine_net).to(device)

#     criterion = nn.L1Loss()
#     optimizer = torch.optim.AdamW(refine_net.parameters(), lr=1e-4)

#     # smoke_test(model, train_loader, criterion, device)

#     num_epochs = 10

#     for epoch in range(num_epochs):
#         train_loss, coarse_l1, refined_l1 = train_one_epoch(
#             model, train_loader, optimizer, criterion, device
#         )

#         print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}")
#         print(f"  coarse L1 : {coarse_l1:.6f}")
#         print(f"  refined L1: {refined_l1:.6f}")
#         print(f"  improvement: {coarse_l1 - refined_l1:.6f}")

#         save_path = f"checkpoints/refine_net_with_depth_epoch_{epoch+1}.pth"
#         torch.save(refine_net.state_dict(), save_path)
#         print(f"Saved refine net checkpoint to {save_path}\n")


# if __name__ == "__main__":
#     main()

import os
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from refine_net import RefineNet
from rvm_with_refine import RVMWithRefine
from model.model import MattingNetwork


class P3M10KDepthDataset(Dataset):
    def __init__(self, root_dir, split="train", image_size=512):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size

        if split == "train":
            self.image_dir = self.root_dir / "train" / "blurred_image"
            self.mask_dir = self.root_dir / "train" / "mask"
            self.depth_dir = self.root_dir / "train" / "depth"
            self.list_path = self.root_dir / "train_list.txt"

        elif split == "val_p":
            self.image_dir = self.root_dir / "validation" / "P3M-500-P" / "blurred_image"
            self.mask_dir = self.root_dir / "validation" / "P3M-500-P" / "mask"
            self.depth_dir = self.root_dir / "validation" / "P3M-500-P" / "depth"
            self.list_path = self.root_dir / "P3M-500-P_list.txt"

        elif split == "val_np":
            self.image_dir = self.root_dir / "validation" / "P3M-500-NP" / "original_image"
            self.mask_dir = self.root_dir / "validation" / "P3M-500-NP" / "mask"
            self.depth_dir = self.root_dir / "validation" / "P3M-500-NP" / "depth"
            self.list_path = self.root_dir / "P3M-500-NP_list.txt"

        else:
            raise ValueError("split must be one of: train, val_p, val_np")

        print(f"\nLoading dataset split={split}")
        print(f"  image_dir = {self.image_dir}")
        print(f"  mask_dir  = {self.mask_dir}")
        print(f"  depth_dir = {self.depth_dir}")
        print(f"  list_path = {self.list_path}")
        print("  image exists:", self.image_dir.exists())
        print("  mask exists :", self.mask_dir.exists())
        print("  depth exists:", self.depth_dir.exists())
        print("  list exists :", self.list_path.exists())

        with open(self.list_path, "r", encoding="utf-8") as f:
            self.names = [line.strip() for line in f if line.strip()]

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.names)

    def _find_file(self, folder, stem):
        candidates = [
            folder / f"{stem}.png",
            folder / f"{stem}.jpg",
            folder / f"{stem}.jpeg",
            folder / stem,
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"Cannot find file for stem {stem} in {folder}")

    def __getitem__(self, idx):
        stem = self.names[idx]

        image_path = self._find_file(self.image_dir, stem)
        mask_path = self._find_file(self.mask_dir, stem)
        depth_path = self._find_file(self.depth_dir, stem)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")

        depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise ValueError(f"Cannot read depth: {depth_path}")

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        image = self.to_tensor(image)  # [3,H,W], [0,1]
        mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0   # [1,H,W]
        depth = torch.from_numpy(depth).float().unsqueeze(0) / 255.0 # [1,H,W]

        return image, mask, depth


def create_dataset(root_dir="P3M-10k", split="train", image_size=512, verbose=True):
    dataset = P3M10KDepthDataset(root_dir=root_dir, split=split, image_size=image_size)

    if verbose:
        print(f"{split} dataset size = {len(dataset)}")
        image, mask, depth = dataset[0]
        print("image shape:", image.shape)
        print("mask shape :", mask.shape)
        print("depth shape:", depth.shape)
        print("image min/max:", image.min().item(), image.max().item())
        print("mask  min/max:", mask.min().item(), mask.max().item())
        print("depth min/max:", depth.min().item(), depth.max().item())

    return dataset


def smoke_test(model, dataloader, criterion, device):
    model.train()
    model.rvm.eval()

    images, alpha_gt, depth = next(iter(dataloader))
    images = images.to(device, non_blocking=True)
    alpha_gt = alpha_gt.to(device, non_blocking=True)
    depth = depth.to(device, non_blocking=True)

    print("\n[Smoke Test with Depth]")
    print("images shape:", images.shape)
    print("alpha_gt shape:", alpha_gt.shape)
    print("depth shape:", depth.shape)

    outputs = model(images, depth, downsample_ratio=0.25)

    pha = outputs["pha"]
    pha_refined = outputs["pha_refined"]
    boundary_mask = outputs["boundary_mask"]
    delta = outputs["delta"]

    print("pha shape:", pha.shape)
    print("pha_refined shape:", pha_refined.shape)
    print("boundary_mask shape:", boundary_mask.shape)
    print("delta shape:", delta.shape)

    loss = criterion(pha_refined, alpha_gt)
    print("loss:", loss.item())

    loss.backward()
    print("backward success!\n")


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    model.rvm.eval()

    total_loss = 0.0
    total_coarse_loss = 0.0
    total_refined_loss = 0.0

    for images, alpha_gt, depth in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        alpha_gt = alpha_gt.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)

        outputs = model(images, depth, downsample_ratio=0.25)

        pha = outputs["pha"]
        pha_refined = outputs["pha_refined"]

        coarse_loss = criterion(pha, alpha_gt)
        refined_loss = criterion(pha_refined, alpha_gt)

        loss = refined_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_coarse_loss += coarse_loss.item()
        total_refined_loss += refined_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_coarse = total_coarse_loss / len(dataloader)
    avg_refined = total_refined_loss / len(dataloader)

    return avg_loss, avg_coarse, avg_refined


def validate_one_epoch(model, dataloader, criterion, device, desc="Validation"):
    model.eval()

    total_loss = 0.0
    total_coarse_loss = 0.0
    total_refined_loss = 0.0

    with torch.no_grad():
        for images, alpha_gt, depth in tqdm(dataloader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            alpha_gt = alpha_gt.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)

            outputs = model(images, depth, downsample_ratio=0.25)

            pha = outputs["pha"]
            pha_refined = outputs["pha_refined"]

            coarse_loss = criterion(pha, alpha_gt)
            refined_loss = criterion(pha_refined, alpha_gt)

            loss = refined_loss

            total_loss += loss.item()
            total_coarse_loss += coarse_loss.item()
            total_refined_loss += refined_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_coarse = total_coarse_loss / len(dataloader)
    avg_refined = total_refined_loss / len(dataloader)

    return avg_loss, avg_coarse, avg_refined


def main():
    os.makedirs("checkpoints", exist_ok=True)

    root_dir = "P3M-10k"
    image_size = 512
    batch_size = 8
    num_workers = 0
    num_epochs = 20
    lr = 1e-4

    train_dataset = create_dataset(root_dir=root_dir, split="train", image_size=image_size, verbose=True)
    val_dataset = create_dataset(root_dir=root_dir, split="val_p", image_size=image_size, verbose=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    rvm_checkpoint_path = "checkpoints/rvm_resnet50.pth"

    rvm = MattingNetwork("resnet50").to(device)
    rvm.load_state_dict(torch.load(rvm_checkpoint_path, map_location=device))

    rvm.eval()
    for param in rvm.parameters():
        param.requires_grad = False

    refine_net = RefineNet(in_channels=5, hidden_channels=32).to(device)
    model = RVMWithRefine(rvm_model=rvm, refine_net=refine_net).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(refine_net.parameters(), lr=lr)

    # smoke_test(model, train_loader, criterion, device)

    best_val_refined = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
        train_loss, train_coarse_l1, train_refined_l1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_coarse_l1, val_refined_l1 = validate_one_epoch(
            model, val_loader, criterion, device, desc="Validation (P)"
        )

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"[Train] loss={train_loss:.6f} | coarse={train_coarse_l1:.6f} | refined={train_refined_l1:.6f} | improvement={train_coarse_l1 - train_refined_l1:.6f}")
        print(f"[Val P] loss={val_loss:.6f} | coarse={val_coarse_l1:.6f} | refined={val_refined_l1:.6f} | improvement={val_coarse_l1 - val_refined_l1:.6f}")

        last_path = f"checkpoints/refine_net_with_depth_epoch_{epoch+1}.pth"
        torch.save(refine_net.state_dict(), last_path)
        print(f"Saved checkpoint to {last_path}")

        if val_refined_l1 < best_val_refined:
            best_val_refined = val_refined_l1
            best_epoch = epoch + 1
            best_path = "checkpoints/refine_net_with_depth_best.pth"
            torch.save(refine_net.state_dict(), best_path)
            print(f"New best checkpoint saved to {best_path}")

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val refined L1: {best_val_refined:.6f}")


if __name__ == "__main__":
    main()