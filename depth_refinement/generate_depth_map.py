import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3


def normalize_depth_to_uint8(depth_np: np.ndarray) -> np.ndarray:
    """
    depth_np: float32 [H, W]
    轉成 0~255 的 uint8 灰階圖
    """
    d_min = depth_np.min()
    d_max = depth_np.max()

    if d_max - d_min < 1e-8:
        return np.zeros_like(depth_np, dtype=np.uint8)

    depth_norm = (depth_np - d_min) / (d_max - d_min)
    depth_u8 = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
    return depth_u8


def read_list(list_path: Path):
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_file(folder: Path, stem: str) -> Path:
    candidates = [
        folder / f"{stem}.jpg",
        folder / f"{stem}.png",
        folder / f"{stem}.jpeg",
        folder / stem,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find file for stem {stem} in {folder}")


class DepthGenerator:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")


        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):

        model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT")
        model = model.to(device=self.device)

        return model

    def predict_depth(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        image_bgr: [H, W, 3], uint8
        return: depth map [H, W], float32
        """

        image_rgb = image_bgr[..., ::-1].copy()

        with torch.no_grad():
            prediction = self.model.inference([image_rgb])

        depth = prediction.depth[0]  # [H, W]

        # 某些實作回來可能已經是 numpy，保險一點都處理
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().float().numpy()
        else:
            depth = np.asarray(depth, dtype=np.float32)

        return depth


def generate_split_depth(root_dir="P3M-10k", split="train"):
    root = Path(root_dir)

    if split == "train":
        image_dir = root / "train" / "blurred_image"
        list_path = root / "train_list.txt"
        save_dir = root / "train" / "depth"
    else:
        pass

    save_dir.mkdir(parents=True, exist_ok=True)

    names = read_list(list_path)
    generator = DepthGenerator()

    for stem in tqdm(names, desc=f"Generating depth for {split}"):
        image_path = find_file(image_dir, stem)
        save_path = save_dir / f"{Path(image_path).stem}.png"

        if save_path.exists():
            continue

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        depth = generator.predict_depth(image_bgr)   # [H, W] float32
        depth_u8 = normalize_depth_to_uint8(depth)

        cv2.imwrite(str(save_path), depth_u8)


if __name__ == "__main__":
    # image_path = "P3M-10k/train/blurred_image/p_00a4eda7.jpg"
    # save_path = "test_depth.png"

    # image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # if image_bgr is None:
    #     raise ValueError(f"Cannot read image: {image_path}")

    # generator = DepthGenerator()
    # depth = generator.predict_depth(image_bgr)
    # depth_u8 = normalize_depth_to_uint8(depth)

    # cv2.imwrite(save_path, depth_u8)

    # print("depth shape:", depth.shape)
    # print("depth min/max:", depth.min(), depth.max())
    # print("saved to:", save_path)
    generate_split_depth(root_dir="P3M-10k", split="train")