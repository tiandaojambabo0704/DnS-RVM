import os
from pathlib import Path

import cv2
import numpy as np
import torch

from refine_net import RefineNet
from rvm_with_refine import RVMWithRefine
from model.model import MattingNetwork


def tensor_from_frame(frame_bgr, device):
    """
    frame_bgr: np.ndarray [H, W, 3], BGR, uint8
    return: torch.Tensor [1, 3, H, W], RGB, float32, [0, 1]
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = frame_rgb.astype(np.float32) / 255.0
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return frame_tensor


def alpha_to_uint8(alpha_tensor):
    """
    alpha_tensor: [1, 1, H, W] or [1, H, W]
    return: uint8 [H, W]
    """
    alpha = alpha_tensor.squeeze().detach().cpu().numpy()
    alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return alpha


def gray_to_bgr(gray_uint8):
    """
    gray_uint8: [H, W]
    return: [H, W, 3]
    """
    return cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)


def composite_with_green_screen(fgr_tensor, alpha_tensor):
    """
    fgr_tensor:   [1, 3, H, W], RGB, [0,1]
    alpha_tensor: [1, 1, H, W], [0,1]

    return:
        comp_bgr: [H, W, 3], uint8, BGR
    """
    fgr = fgr_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()   # HWC RGB
    alpha = alpha_tensor.squeeze().detach().cpu().numpy()[..., None]      # HW1

    green_bg = np.zeros_like(fgr, dtype=np.float32)
    green_bg[..., 1] = 1.0   # RGB = (0, 1, 0)

    comp = fgr * alpha + green_bg * (1.0 - alpha)
    comp = np.clip(comp * 255.0, 0, 255).astype(np.uint8)
    comp_bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
    return comp_bgr


def build_green_compare_canvas(frame_bgr, coarse_green_bgr, refined_green_bgr, boundary_mask_u8):
    """
    2x2 比較畫面:
    [ Input         | Coarse Green  ]
    [ Refined Green | Boundary Mask ]
    """
    boundary_bgr = gray_to_bgr(boundary_mask_u8)

    top = np.concatenate([frame_bgr, coarse_green_bgr], axis=1)
    bottom = np.concatenate([refined_green_bgr, boundary_bgr], axis=1)
    canvas = np.concatenate([top, bottom], axis=0)

    h, w = frame_bgr.shape[:2]

    cv2.putText(canvas, "Input", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(canvas, "Coarse Green", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(canvas, "Refined Green", (10, h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(canvas, "Boundary Mask", (w + 10, h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    return canvas


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================
    # 路徑設定：改這裡就好
    # =========================
    input_video_path = "material.mp4"
    rvm_checkpoint_path = "checkpoints/rvm_resnet50.pth"
    refine_checkpoint_path = "checkpoints/refine_net_epoch_10.pth"

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_alpha_video_path = str(output_dir / "refined_alpha.mp4")
    output_green_video_path = str(output_dir / "refined_green.mp4")
    output_compare_video_path = str(output_dir / "compare_green.mp4")

    # =========================
    # 載入 RVM
    # =========================
    rvm = MattingNetwork("resnet50").to(device)
    rvm_state = torch.load(rvm_checkpoint_path, map_location=device)
    rvm.load_state_dict(rvm_state)
    rvm.eval()

    for p in rvm.parameters():
        p.requires_grad = False

    # =========================
    # 載入 refine net
    # =========================
    refine_net = RefineNet(in_channels=4, hidden_channels=32).to(device)
    refine_state = torch.load(refine_checkpoint_path, map_location=device)
    refine_net.load_state_dict(refine_state)
    refine_net.eval()

    # =========================
    # 組合成完整模型
    # =========================
    model = RVMWithRefine(rvm_model=rvm, refine_net=refine_net).to(device)
    model.eval()

    # =========================
    # 開啟輸入影片
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

    print(f"Video info: {width}x{height}, fps={fps}, total_frames={total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # refined alpha 灰階影片（存成 BGR 三通道）
    writer_alpha = cv2.VideoWriter(
        output_alpha_video_path,
        fourcc,
        fps,
        (width, height)
    )

    # refined 綠幕影片
    writer_green = cv2.VideoWriter(
        output_green_video_path,
        fourcc,
        fps,
        (width, height)
    )

    # 2x2 比較影片
    writer_compare = cv2.VideoWriter(
        output_compare_video_path,
        fourcc,
        fps,
        (width * 2, height * 2)
    )

    # =========================
    # RVM recurrent states
    # 不要每幀重設
    # =========================
    rec = [None, None, None, None]

    frame_idx = 0

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            src = tensor_from_frame(frame_bgr, device)

            outputs = model(src, *rec, downsample_ratio=0.25)
            rec = outputs["rec"]

            fgr = outputs["fgr"]
            pha = outputs["pha"]
            pha_refined = outputs["pha_refined"]
            boundary_mask = outputs["boundary_mask"]

            pha_u8 = alpha_to_uint8(pha)
            pha_refined_u8 = alpha_to_uint8(pha_refined)
            boundary_mask_u8 = alpha_to_uint8(boundary_mask)

            refined_alpha_bgr = gray_to_bgr(pha_refined_u8)

            coarse_green_bgr = composite_with_green_screen(fgr, pha)
            refined_green_bgr = composite_with_green_screen(fgr, pha_refined)

            compare_canvas = build_green_compare_canvas(
                frame_bgr=frame_bgr,
                coarse_green_bgr=coarse_green_bgr,
                refined_green_bgr=refined_green_bgr,
                boundary_mask_u8=boundary_mask_u8
            )

            writer_alpha.write(refined_alpha_bgr)
            writer_green.write(refined_green_bgr)
            writer_compare.write(compare_canvas)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer_alpha.release()
    writer_green.release()
    writer_compare.release()

    print("\nDone!")
    print("Saved refined alpha video to :", output_alpha_video_path)
    print("Saved refined green video to :", output_green_video_path)
    print("Saved compare green video to :", output_compare_video_path)


if __name__ == "__main__":
    main()