"""
Created on Fri Dec 22, 2023
Translated from tf implementation in https://github.com/pat-coady/tiny_imagenet/blob/master/src/vgg_16.py
More details in this blog: https://learningai.io/projects/2017/06/29/tiny-imagenet.html
"""
from collections import OrderedDict
from torch import nn


class VGG16TIN(nn.Module):
    def __init__(self, num_classes=200):
        super(VGG16TIN, self).__init__()
        self.features = nn.Sequential(
            # The dimensions in the comments below assume an input of 56x56
            OrderedDict([
                # (N, 56, 56, 3)
                ('conv1_1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding="same")),
                ('relu1_1', nn.ReLU()),
                ('conv1_2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same")),
                ('relu1_2', nn.ReLU()),
                ('pool1', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),

                # (N, 28, 28, 64)
                ('conv2_1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same")),
                ('relu2_1', nn.ReLU()),
                ('conv2_2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same")),
                ('relu2_2', nn.ReLU()),
                ('pool2', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),

                # (N, 14, 14, 128)
                ('conv3_1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same")),
                ('relu3_1', nn.ReLU()),
                ('conv3_2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding="same")),
                ('relu3_2', nn.ReLU()),
                ('conv3_3', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding="same")),
                ('relu3_3', nn.ReLU()),
                ('pool3', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),

                # (N, 7, 7, 256)
                ('conv4_1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding="same")),
                ('relu4_1', nn.ReLU()),
                ('conv4_2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same")),
                ('relu4_2', nn.ReLU()),
                ('conv4_3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same")),
                ('relu4_3', nn.ReLU()),

                # If the input size is different from 56x56, make sure we have the right dimensions before flattening.
                # This is not a standard layer in the VGG family or architectures.
                ('avgpool', nn.AdaptiveAvgPool2d((7, 7))),

                # fc1: flatten -> fully connected layer
                # (N, 7, 7, 512) -> (N, 25088) -> (N, 4096)
                ('flatten', nn.Flatten()),
                ('fc1', nn.Linear(25088, 4096)),
                ('relu1', nn.ReLU()),
                ('dropout1', nn.Dropout(0.5)),

                # fc2
                # (N, 4096) -> (N, 2048)
                ('fc2', nn.Linear(4096, 2048)),
                ('relu2', nn.ReLU()),
                ('dropout2', nn.Dropout(0.5)),
            ])
        )
        # (N, 2048) -> (N, 200)
        self.logits = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.features(x)
        logits = self.logits(features)
        return logits
