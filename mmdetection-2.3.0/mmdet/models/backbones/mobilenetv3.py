import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import math
import logging
from ..builder import BACKBONES
from .DCNv2.dcn_v2 import DCN

BN_MOMENTUM = 0.1


class DeformConv(nn.Module):

    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.conv = DCN(
            chi,
            cho,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(
                o,
                o,
                f * 2,
                stride=f,
                padding=f // 2,
                output_padding=0,
                groups=o,
                bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


def conv_bn(inp, oup, stride, groups=1, activation=nn.ReLU6):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        activation(inplace=True)
    )


def conv_1x1_bn(inp, oup, groups=1, activation=nn.ReLU6):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        activation(inplace=True)
    )


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + float(3.0), inplace=True) / float(6.0)
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + float(3.0), inplace=True) / float(6.0)
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        self.output_status = False
        if kernel_size == 5 and in_size == 160 and expand_size == 672:
            self.output_status = True
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        if self.output_status:
            expand = out
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out

        if self.output_status:
            return (expand, out)
        return out


@BACKBONES.register_module()
class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, 1000)
        self.init_weights()

        channels = [24, 40, 160, 960]
        # scales = [2**i for i in range(len(channels[self.first_level:]))]
        # self.dla_up = DLAUp(self.first_level, channels[self.first_level:],
        #                     scales)

        out_channel = 24

        self.ida_up = IDAUp(
            out_channel, channels,
            [2**i for i in range(4)])

    # def load_pretrained_layers(self,pretrained):
    #     pretrained_state_dict = torch.load(pretrained)
    #     self.load_state_dict(pretrained_state_dict)
    #     for param in self.parameters():
    #          param.requires_grad = False
    #     print("\nLoaded base model.\n")

    def init_weights(self, pretrained='/home/sdb/wangshentao/myspace/thesis/mbv3_large.old.pth.tar'):  # "./mbv3_large.old.pth.tar"
        if isinstance(pretrained, str):
            logger = logging.getLogger()

            checkpoint = torch.load(pretrained, map_location='cpu')["state_dict"]
            self.load_state_dict(checkpoint, strict=False)
            for param in self.parameters():
                param.requires_grad = True  # to be or not to be

            #  also load module
            # if isinstance(checkpoint, OrderedDict):
            #     state_dict = checkpoint
            # elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            #     state_dict = checkpoint['state_dict']
            # else:
            #     print("No state_dict found in checkpoint file")

            # if list(state_dict.keys())[0].startswith('module.'):
            #     state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
            # # load state_dict
            # if hasattr(self, 'module'):
            #     self.module.load_state_dict( state_dict,strict=False)
            # else:
            #     self.load_state_dict(state_dict,strict=False)

            print("\nLoaded base model.\n")

        elif pretrained is None:
            print("\nNo loaded base model.\n")
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        out = self.hs1(self.bn1(self.conv1(x)))
        extract_idx = [2, 5, 12]
        for i, block in enumerate(self.bneck):
            out = block(out)
            if i in extract_idx:
                outs.append(out)
            if isinstance(out, tuple):
                out = out[1]
        out = self.hs2(self.bn2(self.conv2(out)))
        outs.append(out)
        self.ida_up(outs, 0, len(outs))
        return torch.unsqueeze(outs[-1], 0)


if __name__ == '__main__':
    model = MobileNetV3_Large()
    model.eval()
    input = torch.randn(32, 3, 320, 256)
    y = model(input)
    print(y.size())



