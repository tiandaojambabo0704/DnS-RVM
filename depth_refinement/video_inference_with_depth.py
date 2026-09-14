import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model.model import MattingNetwork
from refine_net import RefineNet
from rvm_with_refine import RVMWithRefine
from depth_anything_3.api import DepthAnything3


def tensor_from_frame(frame_bgr, device):
    """
    frame_bgr: [H, W, 3], BGR uint8
    return: [1, 3, H, W], RGB float32 in [0,1]
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = frame_rgb.astype(np.float32) / 255.0
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return frame_tensor


def alpha_to_uint8(alpha_tensor):
    """
    alpha_tensor: [1,1,H,W] or [1,H,W]
    return: [H,W] uint8
    """
    alpha = alpha_tensor.squeeze().detach().cpu().numpy()
    alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return alpha


def gray_to_bgr(gray_uint8):
    return cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)


def normalize_depth_to_uint8(depth_np: np.ndarray) -> np.ndarray:
    d_min = float(depth_np.min())
    d_max = float(depth_np.max())

    if d_max - d_min < 1e-8:
        return np.zeros_like(depth_np, dtype=np.uint8)

    depth_norm = (depth_np - d_min) / (d_max - d_min)
    depth_u8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    return depth_u8


class DepthGenerator:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")
        model = model.to(device=self.device)
        return model

    def predict_depth(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        image_bgr: [H,W,3] uint8
        return: depth [H,W] float32
        """
        image_rgb = image_bgr[..., ::-1].copy()

        with torch.no_grad():
            prediction = self.model.inference([image_rgb])

        depth = prediction.depth[0]

        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().float().numpy()
        else:
            depth = np.asarray(depth, dtype=np.float32)

        return depth


def depth_gray_to_tensor(depth_gray, device):
    """
    depth_gray: [H,W] uint8
    return: [1,1,H,W] float32 in [0,1]
    """
    depth = depth_gray.astype(np.float32) / 255.0
    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)
    return depth_tensor


def composite_with_green_screen(fgr_tensor, alpha_tensor):
    """
    fgr_tensor:   [1,3,H,W], RGB, [0,1]
    alpha_tensor: [1,1,H,W], [0,1]
    return: [H,W,3] BGR uint8
    """
    fgr = fgr_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()   # RGB
    alpha = alpha_tensor.squeeze().detach().cpu().numpy()[..., None]

    green_bg = np.zeros_like(fgr, dtype=np.float32)
    green_bg[..., 1] = 1.0  # RGB green

    comp = fgr * alpha + green_bg * (1.0 - alpha)
    comp = np.clip(comp * 255.0, 0, 255).astype(np.uint8)
    comp_bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
    return comp_bgr


def build_green_compare_canvas_large(frame_bgr, coarse_green_bgr, refined_green_bgr, boundary_mask_u8, depth_u8):
    """
    2x3 large canvas:
    [ Input         | Coarse Green  | Refined Green ]
    [ Boundary Mask | Depth         | blank         ]
    """
    h, w = frame_bgr.shape[:2]

    boundary_bgr = gray_to_bgr(boundary_mask_u8)
    depth_bgr = gray_to_bgr(depth_u8)
    blank = np.zeros((h, w, 3), dtype=np.uint8)

    top = np.concatenate([frame_bgr, coarse_green_bgr, refined_green_bgr], axis=1)
    bottom = np.concatenate([boundary_bgr, depth_bgr, blank], axis=1)
    canvas = np.concatenate([top, bottom], axis=0)

    cv2.putText(canvas, "Input", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(canvas, "Coarse Green", (w + 10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(canvas, "Refined Green", (2 * w + 10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(canvas, "Boundary Mask", (10, h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(canvas, "Depth", (w + 10, h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    return canvas


def make_even(x: int) -> int:
    return x if x % 2 == 0 else x - 1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================
    # 路徑設定
    # =========================
    input_video_path = "two_women.mp4"
    rvm_checkpoint_path = "checkpoints/rvm_resnet50.pth"
    refine_checkpoint_path = "checkpoints/refine_net_with_depth_best.pth"

    output_dir = Path("outputs_with_depth")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_alpha_video_path = str(output_dir / "refined_alpha.mp4")
    output_green_video_path = str(output_dir / "refined_green.mp4")
    output_green_video_coarse_path = str(output_dir / "coarse_green.mp4")
    output_compare_video_path = str(output_dir / "compare_green.mp4")

    # =========================
    # 載入模型
    # =========================
    rvm = MattingNetwork("resnet50").to(device)
    rvm_state = torch.load(rvm_checkpoint_path, map_location=device)
    rvm.load_state_dict(rvm_state)
    rvm.eval()
    for p in rvm.parameters():
        p.requires_grad = False

    refine_net = RefineNet(in_channels=5, hidden_channels=32).to(device)
    refine_state = torch.load(refine_checkpoint_path, map_location=device)
    refine_net.load_state_dict(refine_state)
    refine_net.eval()

    model = RVMWithRefine(rvm_model=rvm, refine_net=refine_net).to(device)
    model.eval()

    depth_generator = DepthGenerator(device="cuda")

    # =========================
    # 開影片
    # =========================
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = make_even(width)
    height = make_even(height)

    print(f"Video info: {width}x{height}, fps={fps}, total_frames={total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # compare 影片不要存超大畫布，直接存成原影片大小
    compare_out_w = width
    compare_out_h = height

    writer_alpha = cv2.VideoWriter(
        output_alpha_video_path,
        fourcc,
        fps,
        (width, height)
    )

    writer_green = cv2.VideoWriter(
        output_green_video_path,
        fourcc,
        fps,
        (width, height)
    )
    
    writer_coarse_green = cv2.VideoWriter(
        output_green_video_coarse_path,
        fourcc,
        fps,
        (width, height)
    )

    writer_compare = cv2.VideoWriter(
        output_compare_video_path,
        fourcc,
        fps,
        (compare_out_w, compare_out_h)
    )

    print("writer_alpha opened  :", writer_alpha.isOpened())
    print("writer_green opened  :", writer_green.isOpened())
    print("writer_coarse_green opened  :", writer_coarse_green.isOpened())
    print("writer_compare opened:", writer_compare.isOpened())

    if not writer_alpha.isOpened():
        raise RuntimeError("Failed to open writer_alpha")
    if not writer_green.isOpened():
        raise RuntimeError("Failed to open writer_green")
    if not writer_coarse_green.isOpened():
        raise RuntimeError("Failed to open writer_coarse_green")
    if not writer_compare.isOpened():
        raise RuntimeError("Failed to open writer_compare")

    # RVM recurrent states
    rec = [None, None, None, None]

    frame_idx = 0

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # 保險：如果實際讀出尺寸和 metadata 稍有不同，先 resize 一致
            frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

            # 1. RGB tensor
            src = tensor_from_frame(frame_bgr, device)

            # 2. depth
            depth_np = depth_generator.predict_depth(frame_bgr)
            depth_u8 = normalize_depth_to_uint8(depth_np)

            # resize depth 到與 frame 相同大小
            depth_u8 = cv2.resize(depth_u8, (width, height), interpolation=cv2.INTER_LINEAR)
            depth_tensor = depth_gray_to_tensor(depth_u8, device)

            # 3. model forward
            outputs = model(src, depth_tensor, *rec, downsample_ratio=0.25)
            rec = outputs["rec"]

            fgr = outputs["fgr"]
            pha = outputs["pha"]
            pha_refined = outputs["pha_refined"]
            boundary_mask = outputs["boundary_mask"]

            # 4. convert
            pha_refined_u8 = alpha_to_uint8(pha_refined)
            boundary_mask_u8 = alpha_to_uint8(boundary_mask)

            refined_alpha_bgr = gray_to_bgr(pha_refined_u8)
            coarse_green_bgr = composite_with_green_screen(fgr, pha)
            refined_green_bgr = composite_with_green_screen(fgr, pha_refined)

            # 5. build large compare canvas, then resize down
            compare_canvas_large = build_green_compare_canvas_large(
                frame_bgr=frame_bgr,
                coarse_green_bgr=coarse_green_bgr,
                refined_green_bgr=refined_green_bgr,
                boundary_mask_u8=boundary_mask_u8,
                depth_u8=depth_u8
            )

            compare_canvas = cv2.resize(
                compare_canvas_large,
                (compare_out_w, compare_out_h),
                interpolation=cv2.INTER_AREA
            )

            # 6. write
            writer_alpha.write(refined_alpha_bgr)
            writer_green.write(refined_green_bgr)
            writer_compare.write(compare_canvas)
            writer_coarse_green.write(coarse_green_bgr)

            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Processed {frame_idx}/{total_frames}")

    cap.release()
    writer_alpha.release()
    writer_green.release()
    writer_compare.release()
    writer_coarse_green.release()

    print("\nDone!")
    print("Saved refined alpha video to :", output_alpha_video_path)
    print("Saved refined green video to :", output_green_video_path)
    print("Saved coarse green video to :", output_green_video_coarse_path)
    print("Saved compare green video to :", output_compare_video_path)


if __name__ == "__main__":
    main()