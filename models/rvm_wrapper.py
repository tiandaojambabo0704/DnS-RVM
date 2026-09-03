import torch
import numpy as np
import cv2

class OriginalRVM:
    def __init__(self, model_type="resnet50"):  #   mobilenetv3, resnet50
        self.model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            model_type,
            pretrained=True
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.float() / 255.0
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            fgr, pha, *rec = self.model(image_tensor, None, None)

        alpha = pha.squeeze().cpu().numpy()
        alpha = np.clip(alpha, 0, 1)
        return alpha
    
    def load_rvm_model(model_type="resnet50"):
        model = torch.hub.load("PeterL1n/RobustVideoMatting", model_type, pretrained=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        return model, device