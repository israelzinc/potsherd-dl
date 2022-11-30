import torchvision
import torch.nn as nn


def select(model_type: str, num_classes: int):
    if model_type == "resnet18":
        return Resnet18(num_classes)
    elif model_type == "resnet152":
        return Resnet152(num_classes)
    elif model_type == "vgg11":
        return VGG11(num_classes)        
    elif model_type == "efficientnetb0":
        return EfficientNetB0(num_classes)
    elif model_type == "efficientnetb7":
        return EfficientNetB7(num_classes)    
    else:
        raise ValueError(f"Unsupported model type.")


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()
        # self.convnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.convnet = torchvision.models.resnet18(pretrained=True)
        self.convnet.fc = nn.Linear(512, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.convnet(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None


class Resnet152(nn.Module):
    def __init__(self, num_classes):
        super(Resnet152, self).__init__()
        self.convnet = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
        self.convnet.fc = nn.Linear(2048, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.convnet(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None


class VGG11(nn.Module):
    def __init__(self, num_classes):
        super(VGG11, self).__init__()
        self.base = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.IMAGENET1K_V1)
        self.base.classifier[6] = nn.Linear(4096, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB0, self).__init__()
        self.base = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.base.classifier[1] = nn.Linear(1280, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None


class EfficientNetB7(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB7, self).__init__()
        self.base = torchvision.models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.base.classifier[1] = nn.Linear(2560, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None