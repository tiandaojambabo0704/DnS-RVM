import torch
import os
import cv2
import numpy as np 

RGB_SAVE_LIMIT = 1  
ALPHA_BG_MAX = 0.05            
ALPHA_TR_BACK_MAX = 0.5        
ALPHA_TR_FORE_MIN = 0.5       
ALPHA_FORE_MIN = 0.95          

COLOR_MAP = {
    0: [0, 0, 0],          
    1: [255, 0, 0],        
    2: [0, 0, 255],         
    3: [255, 255, 255]      
}

def load_rvm_model(weight_path):
    try:
        model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50") 

        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(weights)
            print(f"RVM success: {weight_path}")
        else:
            print(f"No exist: {weight_path}, use pretrained weight")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        print(f"{device}")

        return model, device
    except Exception as e:
        print(f"{e}")
        return None, None
    
def alpha_to_4class_semantic(alpha_path, save_rgb_path=None, save_label_path=None):
    try:
        alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
        if alpha is None:
            return False, None

        if alpha.max() > 1:
            alpha = alpha.astype(np.float32) / 255.0

        h, w = alpha.shape
        semantic_label = np.zeros((h, w), dtype=np.uint8)

        mask_tr_back = (alpha > ALPHA_BG_MAX) & (alpha <= ALPHA_TR_BACK_MAX)
        semantic_label[mask_tr_back] = 1

        mask_tr_fore = (alpha > ALPHA_TR_FORE_MIN) & (alpha < ALPHA_FORE_MIN)
        semantic_label[mask_tr_fore] = 2

        mask_fore = (alpha >= ALPHA_FORE_MIN)
        semantic_label[mask_fore] = 3

        if save_rgb_path:
            rgb_output = np.zeros((h, w, 3), dtype=np.uint8)
            for label, color in COLOR_MAP.items():
                rgb_output[semantic_label == label] = color
            os.makedirs(os.path.dirname(save_rgb_path), exist_ok=True)
            cv2.imwrite(save_rgb_path, rgb_output)

        if save_label_path:
            os.makedirs(os.path.dirname(save_label_path), exist_ok=True)
            cv2.imwrite(save_label_path, semantic_label)

        return True, semantic_label

    except Exception as e:
        print(f"{e}")
        return False, None
    
def predict_alpha_with_rvm(model, device, img_path, save_alpha_path=None):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False, None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        src = torch.from_numpy(img_rgb).float() / 255.0
        src = src.permute(2, 0, 1).unsqueeze(0)
        src_bgr = src[:, [2, 1, 0], :, :]
        src_bgr = src_bgr.to(device)

        with torch.no_grad():
            fgr, pha = model(src_bgr, None)[:2] 
            alpha = pha[0, 0].cpu().numpy()

        alpha = np.clip(alpha, 0, 1)

        if save_alpha_path:
            os.makedirs(os.path.dirname(save_alpha_path), exist_ok=True)
            alpha_uint8 = (alpha * 255).astype(np.uint8)
            cv2.imwrite(save_alpha_path, alpha_uint8)

        return True, alpha

    except Exception as e:
        print(f"{e}")
        return False, None
    
def process_single_file(args):
    (alpha_path, img_path, sem_rgb_path, sem_label_path, base_alpha_path, idx,
     generate_sem, generate_base, model, device) = args

    results = {"sem_label": False, "base_alpha": False}

    if generate_sem and alpha_path and os.path.exists(alpha_path):
        rgb_path = sem_rgb_path if (idx < RGB_SAVE_LIMIT) else None
        success, _ = alpha_to_4class_semantic(alpha_path, rgb_path, sem_label_path)
        results["sem_label"] = success

    if generate_base and img_path and os.path.exists(img_path) and model is not None:
        success, _ = predict_alpha_with_rvm(model, device, img_path, base_alpha_path)
        results["base_alpha"] = success

    return alpha_path, results