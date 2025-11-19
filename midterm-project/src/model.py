from torchvision import models
import torch.nn as nn

def get_model(pretrained=True):
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    )
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    return model
