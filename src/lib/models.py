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
    elif model_type == "efficientnetb7freeze":
        return EfficientNetB7Freeze(num_classes)    
    elif model_type == "efficientnetb7regl2":
        return EfficientNetB7REGL2(num_classes)    
    elif model_type == "efficientnetb7regl1":
        return EfficientNetB7REGL1(num_classes)            
    elif model_type == "efficientnetb7dropout":
        return EfficientNetB7REGL1(num_classes)            
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
        # self.base = torchvision.models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.base = torchvision.models.efficientnet_b7(pretrained=True)
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



class EfficientNetB7Freeze(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB7Freeze, self).__init__()
        # self.base = torchvision.models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.base = torchvision.models.efficientnet_b7(pretrained=True)
        for param in self.base.parameters():
            
            param.requires_grad = False

        self.base.classifier[1] = nn.Linear(2560, num_classes)
        self.base.classifier[1].require_grads = True

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

class EfficientNetB7REGL2(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB7REGL2, self).__init__()
        # self.base = torchvision.models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.base = torchvision.models.efficientnet_b7(pretrained=True)        
        self.base.classifier[1] = nn.Linear(2560, num_classes)


    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in self.base.parameters()) 
        loss = loss + l2_lambda * l2_norm
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None

class EfficientNetB7REGL1(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB7REGL1, self).__init__()
        # self.base = torchvision.models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.base = torchvision.models.efficientnet_b7(pretrained=True)        
        self.base.classifier[1] = nn.Linear(2560, num_classes)


    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        l2_lambda = 0.001
        l2_norm = sum(p.abs().sum() for p in self.base.parameters()) 
        loss = loss + l2_lambda * l2_norm
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None

class EfficientNetB7Dropout(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(EfficientNetB7, self).__init__()
        self.base = torchvision.models.efficientnet_b7(pretrained=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.base.classifier[1] = nn.Linear(2560, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        outputs = self.dropout(outputs)
        outputs = self.base.classifier[1](outputs)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None
