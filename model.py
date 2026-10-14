import torch
import torch.nn as nn
import torchvision.models as models

class MrBeastScamClassifier(nn.Module):
    def __init__(self, backbone="convnext_tiny", pretrained=True, dropout=0.4):
        super().__init__()
        self.backbone_name = backbone
        
        if backbone == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base_model = models.convnext_tiny(weights=weights)
            in_features = base_model.classifier[2].in_features
            base_model.classifier[2] = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1)
            )
            self.net = base_model
            
        elif backbone == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v3_large(weights=weights)
            in_features = base_model.classifier[0].in_features
            base_model.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(),
                nn.Dropout(p=dropout),
                nn.Linear(256, 1)
            )
            self.net = base_model
            
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base_model = models.resnet34(weights=weights)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1)
            )
            self.net = base_model
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        return logits
        
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
        return probabilities
