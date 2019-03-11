import sys
from torch import nn
import torchvision.models.vgg as models
sys.path.append('../')
import torch.nn.functional as F
from models.networks.network import Net
import math

class caNET(Net):

    def __init__(self, cf, num_classes=1000, pretrained=False, net_name='canet'):
        super(caNET, self).__init__(cf)

        #self.url = 'http://datasets.cvc.uab.es/models/pytorch/basic_vgg16.pth'
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.net_name = net_name

        #self.model = models.vgg16(pretrained=False, num_classes=num_classes)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.features1 = nn.Sequential(
            # Block: conv1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),                 # -30
            nn.ReLU(inplace=True)
        )



        self.features2= nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),  # -28
            nn.ReLU(inplace=True)
        )


        self.features3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),  # -28
            nn.ReLU(inplace=True)
            # -26
        )



        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Conv2d(128, num_classes, kernel_size=1, padding=1),  # -30
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_classes)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        '''if pretrained:
            self.load_basic_weights(net_name)
        else:
            self._initialize_weights()'''

    def forward(self, x):
        x = self.features1(x)
        x = self.maxpool(x)
        x = self.features2(x)
        x = self.maxpool(x)
        x = self.features3(x)
        x = self.maxpool(x)
        x = self.classifier(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), self.num_classes)
        return x

    def _initialize_weights(self):
        pass