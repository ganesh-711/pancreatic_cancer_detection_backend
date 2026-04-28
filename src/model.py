import torch.nn as nn
from torchvision import models


class CNNModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNModel, self).__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last block for fine tuning
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace final FC layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
