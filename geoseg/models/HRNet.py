import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class HRNet(nn.Module):
    def __init__(self, backbone='hrnet_w32', num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        
        # Calculate intermediate channels dynamically
        with torch.no_grad():
            dummy = torch.randn(1, 3, 256, 256)
            out_features = self.backbone(dummy)
            in_channels = sum(f.shape[1] for f in out_features)
            
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        features = self.backbone(x)
        
        # Upsample all representations to the highest resolution (stride 4, features[0])
        target_size = features[0].shape[2:]
        
        out_features = []
        for f in features:
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            out_features.append(f)
            
        out = torch.cat(out_features, dim=1)
        out = self.head(out)
        
        # Final upsampling to the original image size
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out
