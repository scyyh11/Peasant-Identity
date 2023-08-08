import torch
from torch import nn
from torchvision import models
from torchsummary import summary

from tools import mySummary


class CustomizedResNet18(nn.Module):
    def __init__(self):
        super(CustomizedResNet18, self).__init__()
        my_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        my_model.avgpool = nn.AdaptiveAvgPool2d(1)
        my_model.fc = nn.Linear(512, 25)
        self.resnet = my_model

    def forward(self, x):
        return self.resnet(x)


class CustomizedResNet34(nn.Module):
    def __init__(self):
        super(CustomizedResNet34, self).__init__()
        my_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        my_model.avgpool = nn.AdaptiveAvgPool2d(1)
        my_model.fc = nn.Linear(512, 25)
        self.resnet = my_model

    def forward(self, x):
        return self.resnet(x)


class CustomizedResNet50(nn.Module):
    def __init__(self):
        super(CustomizedResNet50, self).__init__()
        my_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        my_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        my_model.fc = nn.Linear(512*4, 25)
        self.resnet = my_model

    def forward(self, x):
        return self.resnet(x)


class CustomizedDensenet121(nn.Module):
    def __init__(self):
        super(CustomizedDensenet121, self).__init__()
        my_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        my_model.classifier = nn.Linear(1024, 25)
        self.densenet = my_model

    def forward(self, x):
        return self.densenet(x)


if __name__ == '__main__':
    model = CustomizedDensenet121()
    # model = CustomizedResNet50()
    mySummary.summary(model, torch.rand(1, 3, 224, 224))
