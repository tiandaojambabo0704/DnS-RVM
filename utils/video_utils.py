import cv2  
import torch    
import numpy as np  
from image_utils import refine_alpha, create_output_frame
import matplotlib.pyplot as plt
import os
import tqdm

def process_single_frame(image_bgr, rvm_model, semantic_model, device,
                         target_size=(512, 512), delta=1.0):
    
    h, w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        fgr, pha, *rec = rvm_model(img_tensor, None, None)
        rvm_alpha = pha.squeeze().cpu().numpy()
        rvm_alpha = np.clip(rvm_alpha, 0, 1)

    img_resized = cv2.resize(image_rgb, target_size)
    base_resized = cv2.resize(rvm_alpha, target_size)

    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    base_tensor = torch.from_numpy(base_resized).unsqueeze(0).unsqueeze(0).float()
    img_tensor = img_tensor.to(device)
    base_tensor = base_tensor.to(device)

    with torch.no_grad():
        semantic_alpha = semantic_model(img_tensor, base_tensor)
        semantic_alpha = semantic_alpha.squeeze().cpu().numpy()
        semantic_alpha = cv2.resize(semantic_alpha, (w, h))
        semantic_alpha = np.clip(semantic_alpha, 0, 1)

    refined_alpha = refine_alpha(rvm_alpha, semantic_alpha, delta=delta)

    return rvm_alpha, semantic_alpha, refined_alpha

def inference_video(input_video_path, output_video_path, rvm_model, semantic_model, device,
                    target_size=(512, 512), delta=1.0, save_frames_dir=None,
                    sample_interval=30, output_type="both"):
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Cannot open: {input_video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video information: {width}x{height}, {fps}fps, {total_frames} frames")

    if output_type == "rvm" or output_type == "refined":
        out_width, out_height = width, height
    elif output_type == "both":
        out_width, out_height = width * 2, height
    elif output_type == "full":
        out_width, out_height = width * 3, height * 2
    else:
        raise ValueError(f"Unknown output_type: {output_type}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)

    frame_count = 0

    with tqdm(total=total_frames, desc="Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rvm_alpha, semantic_alpha, refined_alpha = process_single_frame(
                frame, rvm_model, semantic_model, device, target_size, delta
            )

            output_frame = create_output_frame(frame, rvm_alpha, refined_alpha, output_type)

            out.write(output_frame)

            if save_frames_dir and frame_count % sample_interval == 0 and output_type == "full":
                output_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(18, 12))
                plt.imshow(output_rgb)
                plt.title(f"Frame {frame_count} | Delta={delta}", fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_frames_dir, f"frame_{frame_count:06d}.png"),
                           dpi=150, bbox_inches='tight')
                plt.close()

            frame_count += 1
            pbar.update(1)
            pbar.set_postfix({'frame': frame_count})

    cap.release()
    out.release()
    print(f"   output path: {output_video_path}")
    print(f"  Frame count: {frame_count}")