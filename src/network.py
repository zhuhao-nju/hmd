import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np
import pickle
from torch.nn.modules.loss import _Loss
from mesh_edit import fast_deform_dsa
from renderer import SMPLRenderer
from utility import measure_achr_dist
from torch.autograd import Function
from torchvision.models.resnet import Bottleneck as Bottleneck_ResNet
from torchvision.models.densenet import _DenseBlock as _DenseBlock_DenseNet
from torchvision.models.densenet import _Transition as _Transition_DenseNet
from collections import OrderedDict

# define reshape layer for sequential use        
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(self.shape)

# for VGG network
def make_layers_vgg(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v 
    return nn.Sequential(*layers)

# define the network - vgg16 version
class joint_net_vgg16(nn.Module):
    def __init__(self, 
                 num_classes = 2,
                 in_channels = 2,
                 init_weights = True,
                ):
        super(joint_net_vgg16, self).__init__()
        #cfg_vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
        #       'M', 512, 512, 512, 'M', 512, 512, 512, 'M'] # VGG16
        my_cfg = [64, 64, 128, 128, 'M', 256, 256, 256,
                  'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = make_layers_vgg(my_cfg, 
                                        in_channels, 
                                        batch_norm=True)
        self.linear = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )
        if init_weights:
            self._initialize_weights()
            
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode='fan_out', 
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# define the network - vgg16 version
class anchor_net_vgg16(nn.Module):
    def __init__(self, 
                 num_classes = 1,
                 in_channels = 2,
                 init_weights = True,
                ):
        super(anchor_net_vgg16, self).__init__()
        #cfg_vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
        #       'M', 512, 512, 512, 'M', 512, 512, 512, 'M'] # VGG16
        my_cfg = [64, 64, 128, 128, 'M', 256, 256, 256,
                  512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = make_layers_vgg(my_cfg, 
                                        in_channels, 
                                        batch_norm=True)
        self.linear = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )
        if init_weights:
            self._initialize_weights()
            
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode='fan_out', 
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
class joint_net_res152(nn.Module):
    def __init__(self, 
                 num_classes = 200,
                 in_channels = 4,
                 init_weights = True,
                ):
        self.inplanes = 64
        super(joint_net_res152, self).__init__()

        block = Bottleneck_ResNet
        layers = [3, 8, 36, 3, 3]
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 512, layers[3], stride=2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)        
        
        self.linear = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )
        
        if init_weights:
            self._initialize_weights()
    
    # for resnet
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode='fan_out', 
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
                
class joint_net_dense121(nn.Module):
    def __init__(self, 
                 num_classes = 200,
                 in_channels = 4,
                 init_weights = True,
                ):
        super(joint_net_dense121, self).__init__()
        
        num_init_features = 64
        growth_rate = 32
        block_config = (6, 12, 24, 16)
        drop_rate = 0
        bn_size = 4
        
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock_DenseNet(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition_DenseNet(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        self.linear = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )                
                
        if init_weights:
            self._initialize_weights()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                
                
    def forward(self, x):
        features = self.features(x)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        y = self.linear(out)
        return y
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode='fan_out', 
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Shading net
class shading_net(nn.Module):
    def __init__(self, init_weights=False):
        super(shading_net, self).__init__()
        
        if init_weights:
            self._initialize_weights()
        
        self.enc1 = nn.Sequential(nn.Conv2d(3,16,4,2,1), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(16,32,4,2,1), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(32,64,4,2,1), nn.ReLU(True))
        self.enc4 = nn.Sequential(nn.Conv2d(64,128,4,2,1), nn.ReLU(True))
        self.enc5 = nn.Sequential(nn.Conv2d(128,256,4,2,1), nn.ReLU(True))
        self.encf = nn.Sequential(nn.Conv2d(256,256,3,1,1), nn.ReLU(True))
        self.dec5 = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1), 
                                  nn.ReLU(True))
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(128+128,64,4,2,1), 
                                  nn.ReLU(True))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(64+64,32,4,2,1),
                                  nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(32+32,16,4,2,1),
                                  nn.ReLU(True))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(16+16,16,4,2,1),
                                  nn.ReLU(True))
        self.decf = nn.Sequential(nn.ConvTranspose2d(16,1,3,1,1))
        
    def forward(self, x, mask):
        e1 = self.enc1(x) #16x224x224
        e2 = self.enc2(e1) #32x112x112
        e3 = self.enc3(e2) #64x56x56
        e4 = self.enc4(e3) #128x28x28
        e5 = self.enc5(e4) #256x14x14
        e5 = self.encf(e5) #256x14x14
        d5 = self.dec5(e5) #128x28x28
        d5_skip = torch.cat((e4, d5), dim=1) #256X28X28
        d4 = self.dec4(d5_skip) #64x56x56
        d4_skip = torch.cat((e3, d4), dim=1) #128x56x56
        d3 = self.dec3(d4_skip) #32x112x112
        d3_skip = torch.cat((e2, d3), dim=1) #64x112x112
        d2 = self.dec2(d3_skip) #16x224x224
        d2_skip = torch.cat((e1, d2), dim=1) #32x224x224
        d1 = self.dec1(d2_skip) #16x448x448
        y = self.decf(d1) #1x448X448
        y = y * mask
        return y
        #return x, e1, e2, e3, e4, e5, d5, d4, d3, d2, d1, y
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode='fan_out', 
                                        nonlinearity='relu')
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode='fan_out', 
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                