import os
import sys
import argparse
import torch
from PIL import Image
import torchvision.transforms as T
from model import MrBeastScamClassifier

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_inference_transform():
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_model(checkpoint_path="best_mrbeast_scam_model.pth", device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = checkpoint.get("backbone", "mobilenet_v3_large")
    model = MrBeastScamClassifier(backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device

def predict_image(image_path, model, device, transform=None, threshold=0.5):
    if transform is None:
        transform = get_inference_transform()
        
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image file {image_path}: {e}")
        
    tensor = transform(im).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
            prob = model.predict_proba(tensor).item()
            
    is_scam = prob >= threshold
    return {
        "filepath": image_path,
        "scam_probability": prob,
        "is_scam": is_scam,
        "verdict": "MRBEAST_SCAM" if is_scam else "LEGITIMATE_OR_OTHER",
        "confidence": prob if is_scam else (1.0 - prob)
    }

def main():
    parser = argparse.ArgumentParser(description="Predict if an image is a MrBeast scam.")
    parser.add_argument("images", nargs="+", help="Path(s) to image files to classify")
    parser.add_argument("--model", default="best_mrbeast_scam_model.pth", help="Model checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default 0.5)")
    args = parser.parse_args()
    
    model, device = load_model(args.model)
    print(f"Loaded model on device: {device}\n")
    
    transform = get_inference_transform()
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue
        res = predict_image(img_path, model, device, transform, threshold=args.threshold)
        prob_pct = res["scam_probability"] * 100
        verdict_icon = "🚨" if res["is_scam"] else "✅"
        print(f"{verdict_icon} {res['filepath']}")
        print(f"   Verdict: {res['verdict']}")
        print(f"   Probability: {prob_pct:.2f}% (Confidence: {res['confidence']*100:.2f}%)\n")

if __name__ == "__main__":
    main()
