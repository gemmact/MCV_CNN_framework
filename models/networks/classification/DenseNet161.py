import sys
from torch import nn
import torchvision.models.densenet as models
sys.path.append('../')
from models.networks.network import Net
import math


class DenseNet161(Net):

    def __init__(self, cf, num_classes=21, pretrained=False, net_name='densenet161'):
        super(DenseNet161, self).__init__(cf)

        self.pretrained = pretrained
        self.net_name = net_name

        if self.pretrained:
            self.model = models.densenet161(pretrained = True)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, num_classes)

        self.model = models.densenet161(pretrained=False, num_classes=num_classes)

        '''if pretrained:
            self.load_basic_weights(net_name)
        else:
            self._initialize_weights()'''

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()