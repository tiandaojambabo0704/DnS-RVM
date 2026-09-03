import cv2
import numpy as np 
import matplotlib.pyplot as plt
import torch
import os

from models.semantic_refine_net import SemanticRefineNet
from models.rvm_wrapper import load_rvm_model
from utils.image_utils import refine_alpha, composite_with_bg, create_output_frame
from utils.video_utils import process_single_frame, inference_video
  
def test_single_image(image_path, rvm_model, semantic_model, device, delta=1.0):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot open: {image_path}")
        return

    rvm_alpha, semantic_alpha, refined_alpha = process_single_frame(
        image, rvm_model, semantic_model, device, delta=delta
    )

    rvm_composite = composite_with_bg(image, rvm_alpha)
    refined_composite = composite_with_bg(image, refined_alpha)

    alpha_diff = np.abs(refined_alpha - rvm_alpha)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Input")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(rvm_alpha, cmap='gray')
    axes[0, 1].set_title("RVM Alpha")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(rvm_composite)
    axes[0, 2].set_title("RVM Output")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(refined_alpha, cmap='gray')
    axes[1, 0].set_title(f"Refine Alpha (δ={delta})")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(refined_composite)
    axes[1, 1].set_title("Refine Output")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(alpha_diff, cmap='hot')
    axes[1, 2].set_title("Alpha Diff (RVM - Refine)")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

# test_image_path = "/content/drive/MyDrive/test_image.jpg"
# test_single_image(test_image_path, rvm_model, semantic_model, device, delta=1.0)

def load_your_model(model_path, device):
    model = SemanticRefineNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('val_loss', 'unknown')
        print(f"Load model success. Epoch: {epoch}, Val Loss: {val_loss}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Load model success.")

    model.eval()
    return model

def main():
    INPUT_VIDEO = "/content/drive/MyDrive/DATA/test/input3.mp4"
    OUTPUT_VIDEO = "/content/drive/MyDrive/DATA/test/output3.mp4"
    MODEL_PATH = "/content/drive/MyDrive/RVM/semantic_refine_model/model_best.pth"

    SAVE_FRAMES_DIR = "/content/drive/MyDrive/output_frames"

    OUTPUT_TYPE = "refined"  # "rvm" "refined" "both" "full"

    DELTA = 1.0
    TARGET_SIZE = (512, 512)
    SAMPLE_INTERVAL = 30

    print("=" * 60)
    print("Semantic Refinement")
    print("=" * 60)
    print(f"Input video: {INPUT_VIDEO}")
    print(f"Output video: {OUTPUT_VIDEO}")
    print(f"Refinement δ = {DELTA}")
    print("=" * 60)

    if not os.path.exists(INPUT_VIDEO):
        print(f"Cannot find: {INPUT_VIDEO}")
        return

    rvm_model, device = load_rvm_model(model_type="resnet50")
    semantic_model = load_your_model(MODEL_PATH, device)

    inference_video(
        input_video_path=INPUT_VIDEO,
        output_video_path=OUTPUT_VIDEO,
        rvm_model=rvm_model,
        semantic_model=semantic_model,
        device=device,
        target_size=TARGET_SIZE,
        delta=DELTA,
        save_frames_dir=SAVE_FRAMES_DIR if OUTPUT_TYPE == "full" else None,
        sample_interval=SAMPLE_INTERVAL,
        output_type=OUTPUT_TYPE
    )


if __name__ == "__main__":
    main()