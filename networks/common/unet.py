import torch.nn as nn
import torch
import MinkowskiEngine as ME
from networks.common.resnet_block import ResNetBlock

class UNet(ME.MinkowskiNetwork):
    def __init__(self,
                 in_features=2, 
                 reps=2, 
                 depth=8, 
                 first_num_filters=16, 
                 stride=2, 
                 dropout=0., 
                 input_dropout=0., 
                 scaling='linear', 
                 D=3):
        super(UNet, self).__init__(D)
        
        if scaling == 'exp':
            self.nPlanes = [first_num_filters * (2**i) for i in range(depth)]
        else:
            self.nPlanes = [i * first_num_filters for i in range(1, depth + 1)]
        
        self.input_block = nn.Sequential(
                ME.MinkowskiConvolution(
                in_channels=in_features,
                out_channels=first_num_filters,
                kernel_size=3, stride=1, dimension=D, dilation=1,
                bias=False),
                ME.MinkowskiPReLU(),
                ME.MinkowskiDropout(input_dropout))

        self.encoder = []
        for i, planes in enumerate(self.nPlanes):
            if i < depth - 1:
                m = []
                for _ in range(reps):
                    m.append(ResNetBlock(planes, planes, dimension=D, dropout=dropout))
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=stride, stride=stride, dimension=D, bias=False))
                m.append(ME.MinkowskiBatchNorm(self.nPlanes[i+1], track_running_stats=True))
                m.append(ME.MinkowskiPReLU())
                m = nn.Sequential(*m)
                self.encoder.append(m)
        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = []
        # reverse order
        for i, planes in reversed(list(enumerate(self.nPlanes))):
            if i > 0:
                m = []
                for _ in range(reps):
                    m.append(ResNetBlock(planes, planes, dimension=D, dropout=dropout))
                m.append(ME.MinkowskiConvolutionTranspose(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i-1],
                    kernel_size=stride, stride=stride, dimension=D, bias=False))
                m.append(ME.MinkowskiBatchNorm(self.nPlanes[i-1], track_running_stats=True))
                m.append(ME.MinkowskiPReLU())
                m = nn.Sequential(*m)
                self.decoder.append(m)
        self.decoder = nn.Sequential(*self.decoder)
        
    def forward(self, x):
        x = self.input_block(x)
        enc_feature_maps = [x]
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            if i < len(self.encoder) - 1:
                enc_feature_maps.insert(0, x)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
            x = x + enc_feature_maps[i]
        return x