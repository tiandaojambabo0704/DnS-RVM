import torch
import cv2
import numpy as np

def refine_alpha(rvm_alpha, semantic_alpha, delta=0.7, transition_range=(0.05, 0.95)):
    low, high = transition_range
    weight = np.zeros_like(rvm_alpha)
    transition_mask = (rvm_alpha > low) & (rvm_alpha < high)
    weight[transition_mask] = delta
    weight = cv2.GaussianBlur(weight, (5, 5), 1.0)
    refined = (1 - weight) * rvm_alpha + weight * semantic_alpha
    refined = np.clip(refined, 0, 1)
    return refined

def composite_with_green_bg(image, alpha, bg_color=(0, 255, 0)):
    h, w = image.shape[:2]
    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h))

    bg = np.full((h, w, 3), bg_color, dtype=np.uint8)
    alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    composite = (image_rgb * alpha_3ch + bg * (1 - alpha_3ch)).astype(np.uint8)
    return composite

def composite_with_bg(image_bgr, alpha, bg_color=(0, 255, 0)):
    h, w = image_bgr.shape[:2]
    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h))

    bg = np.full((h, w, 3), bg_color, dtype=np.uint8)
    alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)

    composite = (image_bgr * alpha_3ch + bg * (1 - alpha_3ch)).astype(np.uint8)
    return composite

def create_output_frame(image_bgr, rvm_alpha, refined_alpha, output_type="both"):
    h, w = image_bgr.shape[:2]

    if output_type == "rvm":
        rvm_composite = composite_with_bg(image_bgr, rvm_alpha)
        return rvm_composite

    elif output_type == "refined":
        refined_composite = composite_with_bg(image_bgr, refined_alpha)
        return refined_composite

    elif output_type == "both":
        rvm_composite = composite_with_bg(image_bgr, rvm_alpha)
        refined_composite = composite_with_bg(image_bgr, refined_alpha)
        return np.hstack([rvm_composite, refined_composite])

    elif output_type == "full":
        input_frame = image_bgr 

        rvm_alpha_vis = (rvm_alpha * 255).astype(np.uint8)
        rvm_alpha_color = cv2.cvtColor(rvm_alpha_vis, cv2.COLOR_GRAY2BGR)
        rvm_composite = composite_with_bg(image_bgr, rvm_alpha)

        refined_alpha_vis = (refined_alpha * 255).astype(np.uint8)
        refined_alpha_color = cv2.cvtColor(refined_alpha_vis, cv2.COLOR_GRAY2BGR)
        refined_composite = composite_with_bg(image_bgr, refined_alpha)

        alpha_diff = np.abs(refined_alpha - rvm_alpha)
        alpha_diff_vis = (alpha_diff * 255).astype(np.uint8)
        alpha_diff_color = cv2.applyColorMap(alpha_diff_vis, cv2.COLORMAP_JET)

        row1 = np.hstack([input_frame, rvm_alpha_color, rvm_composite])
        row2 = np.hstack([refined_alpha_color, refined_composite, alpha_diff_color])
        return np.vstack([row1, row2])

    else:
        raise ValueError(f"Unknown output_type: {output_type}")