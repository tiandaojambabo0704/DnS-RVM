import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from model.model import MattingNetwork
from refine_net import RefineNet
from rvm_with_refine import RVMWithRefine


class P3MValidationDepthDataset(Dataset):
    """
    Inference still uses 512x512 input because the depth-refinement network was
    trained/evaluated with fixed 512 inputs.

    Metrics are computed after upsampling predicted alpha back to the original
    image size and comparing with the original-size ground-truth mask.
    """
    def __init__(self, root_dir="P3M-10k", subset="P3M-500-P", image_size=512):
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.image_size = image_size

        if subset == "P3M-500-P":
            self.image_dir = self.root_dir / "validation" / "P3M-500-P" / "blurred_image"
            self.mask_dir = self.root_dir / "validation" / "P3M-500-P" / "mask"
            self.depth_dir = self.root_dir / "validation" / "P3M-500-P" / "depth"
            self.list_path = self.root_dir / "P3M-500-P_list.txt"
        elif subset == "P3M-500-NP":
            self.image_dir = self.root_dir / "validation" / "P3M-500-NP" / "original_image"
            self.mask_dir = self.root_dir / "validation" / "P3M-500-NP" / "mask"
            self.depth_dir = self.root_dir / "validation" / "P3M-500-NP" / "depth"
            self.list_path = self.root_dir / "P3M-500-NP_list.txt"
        else:
            raise ValueError("subset must be 'P3M-500-P' or 'P3M-500-NP'")

        with open(self.list_path, "r", encoding="utf-8") as f:
            self.names = [line.strip() for line in f if line.strip()]

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.names)

    def _find_file(self, folder: Path, stem: str) -> Path:
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

        name = Path(image_path).stem
        original_h, original_w = image.shape[:2]

        # Keep original-size GT for final metric computation.
        mask_original_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        # Resize input/depth/mask to 512 for model inference and 512 canvas.
        image_512 = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask_512 = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        depth_512 = cv2.resize(
            depth,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        image_tensor = self.to_tensor(image_512)
        mask_512_tensor = torch.from_numpy(mask_512).float().unsqueeze(0) / 255.0
        depth_tensor = torch.from_numpy(depth_512).float().unsqueeze(0) / 255.0

        return (
            image_tensor,
            mask_512_tensor,
            mask_original_tensor,
            depth_tensor,
            name,
            original_h,
            original_w,
        )


def tensor_to_uint8_gray(x):
    if x.ndim == 3:
        x = x.squeeze(0)
    x = x.detach().cpu().numpy()
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x


def tensor_to_uint8_rgb(x):
    x = x.detach().cpu().permute(1, 2, 0).numpy()
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x


def gray_to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def save_gray(path, gray):
    cv2.imwrite(str(path), gray)


def save_rgb(path, rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def resize_rgb_to_original(rgb, original_w, original_h):
    return cv2.resize(rgb, (original_w, original_h), interpolation=cv2.INTER_LINEAR)


def resize_gray_to_original(gray, original_w, original_h):
    return cv2.resize(gray, (original_w, original_h), interpolation=cv2.INTER_LINEAR)


def upsample_alpha_to_original(alpha, original_h, original_w):
    """
    alpha: [1, 1, 512, 512]
    return: [1, 1, original_h, original_w]
    """
    return F.interpolate(
        alpha,
        size=(original_h, original_w),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)


def composite_with_green_screen(fgr_tensor, alpha_tensor):
    """
    fgr_tensor: [1, 3, H, W]
    alpha_tensor: [1, 1, H, W]
    return: uint8 RGB image [H, W, 3]
    """
    fgr = fgr_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    alpha = alpha_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()[..., None]

    green_bg = np.zeros_like(fgr, dtype=np.float32)
    green_bg[..., 1] = 1.0

    comp = fgr * alpha + green_bg * (1.0 - alpha)
    comp = np.clip(comp * 255.0, 0, 255).astype(np.uint8)
    return comp


def build_compare_canvas(image_rgb, coarse_gray, refined_gray, gt_gray, diff_gray, depth_gray):
    """
    2x3 canvas at 512x512 scale:
    Input | Coarse | Refined
    GT    | Diff   | Depth
    """
    h, w = coarse_gray.shape

    input_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    coarse_bgr = gray_to_bgr(coarse_gray)
    refined_bgr = gray_to_bgr(refined_gray)
    gt_bgr = gray_to_bgr(gt_gray)
    diff_bgr = gray_to_bgr(diff_gray)
    depth_bgr = gray_to_bgr(depth_gray)

    top = np.concatenate([input_bgr, coarse_bgr, refined_bgr], axis=1)
    bottom = np.concatenate([gt_bgr, diff_bgr, depth_bgr], axis=1)
    canvas = np.concatenate([top, bottom], axis=0)

    cv2.putText(canvas, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(canvas, "Coarse", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(canvas, "Refined", (2 * w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(canvas, "GT", (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(canvas, "|Refined-Coarse|", (w + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(canvas, "Depth", (2 * w + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    return canvas


# =========================
# Full-image metrics
# =========================

def compute_l1(pred, gt):
    return torch.mean(torch.abs(pred - gt)).item()


def compute_mse(pred, gt):
    return torch.mean((pred - gt) ** 2).item()


def compute_sad(pred, gt):
    return torch.sum(torch.abs(pred - gt)).item()


# =========================
# Transition-region metrics
# =========================

def make_transition_mask(alpha, low=0.05, high=0.95):
    """
    Use ground-truth alpha to define transition region:
        low < gt_alpha < high
    """
    return (alpha > low) & (alpha < high)


def compute_transition_metrics(pred, gt, low=0.05, high=0.95):
    mask = make_transition_mask(gt, low=low, high=high)
    transition_pixels = int(mask.sum().item())

    if transition_pixels == 0:
        return {
            "l1": float("nan"),
            "mse": float("nan"),
            "sad": 0.0,
            "pixels": 0,
        }

    diff = pred[mask] - gt[mask]
    abs_diff = torch.abs(diff)

    return {
        "l1": torch.mean(abs_diff).item(),
        "mse": torch.mean(diff ** 2).item(),
        "sad": torch.sum(abs_diff).item(),
        "pixels": transition_pixels,
    }


def safe_average(values):
    clean_values = []
    for v in values:
        if not np.isnan(v):
            clean_values.append(v)
    if len(clean_values) == 0:
        return float("nan")
    return sum(clean_values) / len(clean_values)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================
    # Paths and settings
    # =========================
    p3m_root = "P3M-10k"
    subset = "P3M-500-NP"  # Change to "P3M-500-P" if needed.

    rvm_ckpt = "checkpoints/rvm_resnet50.pth"
    refine_ckpt = "checkpoints/refine_net_with_depth_best.pth"

    image_size = 512
    downsample_ratio = 0.25

    transition_low = 0.05
    transition_high = 0.95

    save_root = Path(f"compare_outputs_with_depth_{subset}_eval_original_size")
    save_root.mkdir(exist_ok=True)

    # =========================
    # Save folders
    # =========================
    save_input = save_root / "input_original_size"
    save_coarse = save_root / "coarse_alpha_original_size"
    save_refined = save_root / "refined_alpha_original_size"
    save_gt = save_root / "gt_alpha_original_size"
    save_depth = save_root / "depth_original_size"
    save_diff = save_root / "diff_map_original_size"
    save_green_coarse = save_root / "green_coarse_original_size"
    save_green_refined = save_root / "green_refined_original_size"
    save_canvas = save_root / "canvas_512"

    for folder in [
        save_input,
        save_coarse,
        save_refined,
        save_gt,
        save_depth,
        save_diff,
        save_green_coarse,
        save_green_refined,
        save_canvas,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    # =========================
    # Dataset
    # =========================
    dataset = P3MValidationDepthDataset(
        root_dir=p3m_root,
        subset=subset,
        image_size=image_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # =========================
    # Load RVM
    # =========================
    rvm = MattingNetwork("resnet50").to(device)
    rvm.load_state_dict(torch.load(rvm_ckpt, map_location=device))
    rvm.eval()

    for p in rvm.parameters():
        p.requires_grad = False

    # =========================
    # Load Depth RefineNet
    # =========================
    refine_net = RefineNet(in_channels=5, hidden_channels=32).to(device)
    refine_net.load_state_dict(torch.load(refine_ckpt, map_location=device))
    refine_net.eval()

    model = RVMWithRefine(rvm_model=rvm, refine_net=refine_net).to(device)
    model.eval()

    # =========================
    # Metric lists
    # =========================
    coarse_l1_list = []
    refined_l1_list = []
    coarse_mse_list = []
    refined_mse_list = []
    coarse_sad_list = []
    refined_sad_list = []

    coarse_transition_l1_list = []
    refined_transition_l1_list = []
    coarse_transition_mse_list = []
    refined_transition_mse_list = []
    coarse_transition_sad_list = []
    refined_transition_sad_list = []

    transition_pixel_list = []
    per_image_rows = []

    # =========================
    # Inference and evaluation
    # =========================
    with torch.no_grad():
        for (
            image,
            alpha_gt_512,
            alpha_gt_original,
            depth,
            name,
            original_h,
            original_w,
        ) in tqdm(dataloader, desc="Comparing with depth"):
            image = image.to(device)                        # [1, 3, 512, 512]
            alpha_gt_512 = alpha_gt_512.to(device)          # [1, 1, 512, 512], for canvas only
            alpha_gt_original = alpha_gt_original.to(device)  # [1, 1, original_h, original_w]
            depth = depth.to(device)                        # [1, 1, 512, 512]
            name = name[0]

            original_h = int(original_h.item())
            original_w = int(original_w.item())

            outputs = model(
                src=image,
                depth=depth,
                downsample_ratio=downsample_ratio,
            )

            fgr = outputs["fgr"]
            pha = outputs["pha"].clamp(0.0, 1.0)                    # [1, 1, 512, 512]
            pha_refined = outputs["pha_refined"].clamp(0.0, 1.0)    # [1, 1, 512, 512]

            # =========================
            # Original-size metrics
            # =========================
            pha_original = upsample_alpha_to_original(pha, original_h, original_w)
            pha_refined_original = upsample_alpha_to_original(pha_refined, original_h, original_w)

            coarse_l1 = compute_l1(pha_original, alpha_gt_original)
            refined_l1 = compute_l1(pha_refined_original, alpha_gt_original)

            coarse_mse = compute_mse(pha_original, alpha_gt_original)
            refined_mse = compute_mse(pha_refined_original, alpha_gt_original)

            coarse_sad = compute_sad(pha_original, alpha_gt_original)
            refined_sad = compute_sad(pha_refined_original, alpha_gt_original)

            coarse_trans = compute_transition_metrics(
                pha_original,
                alpha_gt_original,
                low=transition_low,
                high=transition_high,
            )
            refined_trans = compute_transition_metrics(
                pha_refined_original,
                alpha_gt_original,
                low=transition_low,
                high=transition_high,
            )

            coarse_l1_list.append(coarse_l1)
            refined_l1_list.append(refined_l1)
            coarse_mse_list.append(coarse_mse)
            refined_mse_list.append(refined_mse)
            coarse_sad_list.append(coarse_sad)
            refined_sad_list.append(refined_sad)

            coarse_transition_l1_list.append(coarse_trans["l1"])
            refined_transition_l1_list.append(refined_trans["l1"])
            coarse_transition_mse_list.append(coarse_trans["mse"])
            refined_transition_mse_list.append(refined_trans["mse"])
            coarse_transition_sad_list.append(coarse_trans["sad"])
            refined_transition_sad_list.append(refined_trans["sad"])
            transition_pixel_list.append(coarse_trans["pixels"])

            per_image_rows.append({
                "name": name,
                "original_h": original_h,
                "original_w": original_w,
                "eval_size": f"{original_h}x{original_w}",
                "model_input_size": f"{image_size}x{image_size}",

                "coarse_l1": coarse_l1,
                "refined_l1": refined_l1,
                "l1_improvement": coarse_l1 - refined_l1,

                "coarse_mse": coarse_mse,
                "refined_mse": refined_mse,
                "mse_improvement": coarse_mse - refined_mse,

                "coarse_sad": coarse_sad,
                "refined_sad": refined_sad,
                "sad_improvement": coarse_sad - refined_sad,

                "transition_pixels": coarse_trans["pixels"],

                "coarse_transition_l1": coarse_trans["l1"],
                "refined_transition_l1": refined_trans["l1"],
                "transition_l1_improvement": coarse_trans["l1"] - refined_trans["l1"],

                "coarse_transition_mse": coarse_trans["mse"],
                "refined_transition_mse": refined_trans["mse"],
                "transition_mse_improvement": coarse_trans["mse"] - refined_trans["mse"],

                "coarse_transition_sad": coarse_trans["sad"],
                "refined_transition_sad": refined_trans["sad"],
                "transition_sad_improvement": coarse_trans["sad"] - refined_trans["sad"],
            })

            # =========================
            # 512 canvas images
            # =========================
            image_rgb_512 = tensor_to_uint8_rgb(image[0])
            coarse_gray_512 = tensor_to_uint8_gray(pha[0])
            refined_gray_512 = tensor_to_uint8_gray(pha_refined[0])
            gt_gray_512 = tensor_to_uint8_gray(alpha_gt_512[0])
            depth_gray_512 = tensor_to_uint8_gray(depth[0])

            diff_512 = torch.abs(pha_refined - pha)
            diff_gray_512 = tensor_to_uint8_gray(diff_512[0])

            canvas = build_compare_canvas(
                image_rgb=image_rgb_512,
                coarse_gray=coarse_gray_512,
                refined_gray=refined_gray_512,
                gt_gray=gt_gray_512,
                diff_gray=diff_gray_512,
                depth_gray=depth_gray_512,
            )

            # =========================
            # Save original-size outputs
            # =========================
            image_rgb_save = resize_rgb_to_original(image_rgb_512, original_w, original_h)
            coarse_gray_save = tensor_to_uint8_gray(pha_original[0])
            refined_gray_save = tensor_to_uint8_gray(pha_refined_original[0])
            gt_gray_save = tensor_to_uint8_gray(alpha_gt_original[0])
            depth_gray_save = resize_gray_to_original(depth_gray_512, original_w, original_h)
            diff_gray_save = np.abs(refined_gray_save.astype(np.int16) - coarse_gray_save.astype(np.int16))
            diff_gray_save = np.clip(diff_gray_save, 0, 255).astype(np.uint8)

            # For green-screen visualization, upsample fgr to original size too.
            fgr_original = F.interpolate(
                fgr,
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            green_coarse_rgb_save = composite_with_green_screen(fgr_original, pha_original)
            green_refined_rgb_save = composite_with_green_screen(fgr_original, pha_refined_original)

            save_rgb(save_input / f"{name}.png", image_rgb_save)
            save_gray(save_coarse / f"{name}.png", coarse_gray_save)
            save_gray(save_refined / f"{name}.png", refined_gray_save)
            save_gray(save_gt / f"{name}.png", gt_gray_save)
            save_gray(save_depth / f"{name}.png", depth_gray_save)
            save_gray(save_diff / f"{name}.png", diff_gray_save)
            save_rgb(save_green_coarse / f"{name}.png", green_coarse_rgb_save)
            save_rgb(save_green_refined / f"{name}.png", green_refined_rgb_save)
            cv2.imwrite(str(save_canvas / f"{name}.png"), canvas)

            print(
                f"{name} | "
                f"eval original size: {original_h}x{original_w} | "
                f"Full MSE coarse/refined: {coarse_mse:.6f}/{refined_mse:.6f} | "
                f"Full SAD coarse/refined: {coarse_sad:.2f}/{refined_sad:.2f} | "
                f"Trans MSE coarse/refined: {coarse_trans['mse']:.6f}/{refined_trans['mse']:.6f} | "
                f"Trans SAD coarse/refined: {coarse_trans['sad']:.2f}/{refined_trans['sad']:.2f}"
            )

    # =========================
    # Summary
    # =========================
    avg_coarse_l1 = safe_average(coarse_l1_list)
    avg_refined_l1 = safe_average(refined_l1_list)

    avg_coarse_mse = safe_average(coarse_mse_list)
    avg_refined_mse = safe_average(refined_mse_list)

    avg_coarse_sad = safe_average(coarse_sad_list)
    avg_refined_sad = safe_average(refined_sad_list)

    avg_coarse_transition_l1 = safe_average(coarse_transition_l1_list)
    avg_refined_transition_l1 = safe_average(refined_transition_l1_list)

    avg_coarse_transition_mse = safe_average(coarse_transition_mse_list)
    avg_refined_transition_mse = safe_average(refined_transition_mse_list)

    avg_coarse_transition_sad = safe_average(coarse_transition_sad_list)
    avg_refined_transition_sad = safe_average(refined_transition_sad_list)

    avg_transition_pixels = safe_average(transition_pixel_list)

    print("\n===== Summary: Full Image Metrics, evaluated at original image size =====")
    print(f"Average coarse L1      : {avg_coarse_l1:.6f}")
    print(f"Average refined L1     : {avg_refined_l1:.6f}")
    print(f"L1 improvement         : {avg_coarse_l1 - avg_refined_l1:.6f}")

    print(f"Average coarse MSE     : {avg_coarse_mse:.6f}")
    print(f"Average refined MSE    : {avg_refined_mse:.6f}")
    print(f"MSE improvement        : {avg_coarse_mse - avg_refined_mse:.6f}")

    print(f"Average coarse SAD     : {avg_coarse_sad:.2f}")
    print(f"Average refined SAD    : {avg_refined_sad:.2f}")
    print(f"SAD improvement        : {avg_coarse_sad - avg_refined_sad:.2f}")

    print("\n===== Summary: Transition Region Metrics, evaluated at original image size =====")
    print(f"Transition range       : {transition_low} < gt_alpha < {transition_high}")
    print(f"Average trans pixels   : {avg_transition_pixels:.2f}")

    print(f"Average coarse trans L1 : {avg_coarse_transition_l1:.6f}")
    print(f"Average refined trans L1: {avg_refined_transition_l1:.6f}")
    print(f"Trans L1 improvement    : {avg_coarse_transition_l1 - avg_refined_transition_l1:.6f}")

    print(f"Average coarse trans MSE : {avg_coarse_transition_mse:.6f}")
    print(f"Average refined trans MSE: {avg_refined_transition_mse:.6f}")
    print(f"Trans MSE improvement    : {avg_coarse_transition_mse - avg_refined_transition_mse:.6f}")

    print(f"Average coarse trans SAD : {avg_coarse_transition_sad:.2f}")
    print(f"Average refined trans SAD: {avg_refined_transition_sad:.2f}")
    print(f"Trans SAD improvement    : {avg_coarse_transition_sad - avg_refined_transition_sad:.2f}")

    # =========================
    # Save metrics.txt
    # =========================
    metrics_path = save_root / "metrics.txt"

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Subset: {subset}\n")
        f.write("Evaluation size: original image size\n")
        f.write(f"Model input size: {image_size}x{image_size}\n")
        f.write("Predicted alpha: upsampled from 512x512 to original size before metrics\n")
        f.write("Saved individual outputs: original image size\n")
        f.write("Saved canvas: 512x512-based comparison canvas\n")
        f.write(f"Downsample ratio: {downsample_ratio}\n")
        f.write(f"Transition region: {transition_low} < gt_alpha < {transition_high}\n\n")

        f.write("===== Full Image Metrics =====\n")
        f.write(f"Average coarse L1      : {avg_coarse_l1:.6f}\n")
        f.write(f"Average refined L1     : {avg_refined_l1:.6f}\n")
        f.write(f"L1 improvement         : {avg_coarse_l1 - avg_refined_l1:.6f}\n\n")

        f.write(f"Average coarse MSE     : {avg_coarse_mse:.6f}\n")
        f.write(f"Average refined MSE    : {avg_refined_mse:.6f}\n")
        f.write(f"MSE improvement        : {avg_coarse_mse - avg_refined_mse:.6f}\n\n")

        f.write(f"Average coarse SAD     : {avg_coarse_sad:.2f}\n")
        f.write(f"Average refined SAD    : {avg_refined_sad:.2f}\n")
        f.write(f"SAD improvement        : {avg_coarse_sad - avg_refined_sad:.2f}\n\n")

        f.write("===== Transition Region Metrics =====\n")
        f.write(f"Transition range       : {transition_low} < gt_alpha < {transition_high}\n")
        f.write(f"Average trans pixels   : {avg_transition_pixels:.2f}\n\n")

        f.write(f"Average coarse transition L1      : {avg_coarse_transition_l1:.6f}\n")
        f.write(f"Average refined transition L1     : {avg_refined_transition_l1:.6f}\n")
        f.write(f"Transition L1 improvement         : {avg_coarse_transition_l1 - avg_refined_transition_l1:.6f}\n\n")

        f.write(f"Average coarse transition MSE     : {avg_coarse_transition_mse:.6f}\n")
        f.write(f"Average refined transition MSE    : {avg_refined_transition_mse:.6f}\n")
        f.write(f"Transition MSE improvement        : {avg_coarse_transition_mse - avg_refined_transition_mse:.6f}\n\n")

        f.write(f"Average coarse transition SAD     : {avg_coarse_transition_sad:.2f}\n")
        f.write(f"Average refined transition SAD    : {avg_refined_transition_sad:.2f}\n")
        f.write(f"Transition SAD improvement        : {avg_coarse_transition_sad - avg_refined_transition_sad:.2f}\n")

    # =========================
    # Save per-image CSV
    # =========================
    csv_path = save_root / "per_image_metrics.csv"

    if len(per_image_rows) > 0:
        fieldnames = list(per_image_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_image_rows)

    print(f"\nSaved results to: {save_root}")
    print(f"Saved summary metrics to: {metrics_path}")
    print(f"Saved per-image metrics to: {csv_path}")


if __name__ == "__main__":
    main()
