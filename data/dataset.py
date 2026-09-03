from torch.utils.data import Dataset
import os
import glob
import cv2
import torch
import numpy as np

class RefineDataset(Dataset):
    def __init__(self, data_root, split='train', size=(512, 512),
                 use_confidence_weight=True):
        self.size = size
        self.use_confidence_weight = use_confidence_weight

        if split == 'train':
            self.origin_dir = os.path.join(data_root, 'train', 'blurred_image')
            self.mask_dir = os.path.join(data_root, 'train', 'mask')
            self.base_alpha_dir = os.path.join(data_root, 'train', 'base_alpha')
            self.sem_label_dir = os.path.join(data_root, 'train', 'sem_label')
        else:
            self.origin_dir = os.path.join(data_root, 'validation', 'P3M-500-P', 'blurred_image')
            self.mask_dir = os.path.join(data_root, 'validation', 'P3M-500-P', 'mask')
            self.base_alpha_dir = os.path.join(data_root, 'validation', 'P3M-500-P', 'base_alpha')
            self.sem_label_dir = os.path.join(data_root, 'validation', 'P3M-500-P', 'sem_label')

        self.image_paths = sorted(glob.glob(os.path.join(self.origin_dir, '*.*')))

        valid_paths = []
        for img_path in self.image_paths:
            name = os.path.basename(img_path).split('.')[0]

            mask_path = os.path.join(self.mask_dir, f"{name}.png")
            base_path = os.path.join(self.base_alpha_dir, f"{name}.png")

            if os.path.exists(mask_path) and os.path.exists(base_path):
                valid_paths.append(img_path)

        self.image_paths = valid_paths
        print(f" {split} dataset: {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        name = os.path.basename(img_path).split('.')[0]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        mask_path = os.path.join(self.mask_dir, f"{name}.png")
        gt_alpha = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_alpha = cv2.resize(gt_alpha, self.size)
        gt_alpha = gt_alpha.astype(np.float32) / 255.0
        gt_alpha = torch.from_numpy(gt_alpha).unsqueeze(0)

        base_path = os.path.join(self.base_alpha_dir, f"{name}.png")
        base_alpha = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
        base_alpha = cv2.resize(base_alpha, self.size)
        base_alpha = base_alpha.astype(np.float32) / 255.0
        base_alpha = torch.from_numpy(base_alpha).unsqueeze(0)

        sem_label = None
        if self.use_confidence_weight:
            sem_path = os.path.join(self.sem_label_dir, f"{name}.png")
            if os.path.exists(sem_path):
                sem_label = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
                sem_label = cv2.resize(sem_label, self.size, interpolation=cv2.INTER_NEAREST)
                sem_label = torch.from_numpy(sem_label).long()
            else:
                alpha_np = gt_alpha.squeeze(0).numpy()
                sem_label = np.zeros_like(alpha_np, dtype=np.int64)
                sem_label[(alpha_np > 0.05) & (alpha_np <= 0.5)] = 1
                sem_label[(alpha_np > 0.5) & (alpha_np < 0.95)] = 2
                sem_label[alpha_np >= 0.95] = 3
                sem_label = torch.from_numpy(sem_label).long()

        result = {
            'image': image,
            'gt_alpha': gt_alpha,
            'base_alpha': base_alpha,
            'name': name
        }

        if sem_label is not None:
            result['sem_label'] = sem_label

        return result