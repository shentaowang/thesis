import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, caffe2_xavier_init
from torch.utils.checkpoint import checkpoint

from ..builder import NECKS


@NECKS.register_module()
class NOFPN(nn.Module):
    """
    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        stride (int): stride of 3x3 convolutional layers
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 pooling_type='AVG',
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cp=False,
                 stride=1):
        super(NOFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.fpn_convs = nn.ModuleList()
        # self.fpn_convs.append(
        #     ConvModule(
        #         in_channels[0],
        #         out_channels,
        #         kernel_size=3,
        #         padding=1,
        #         stride=stride,
        #         conv_cfg=self.conv_cfg,
        #         act_cfg=None))

        if pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

    def init_weights(self):
        """Initialize the weights of module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):
        """Forward function."""
        # assert len(inputs) == self.num_ins

        outputs = []
        outputs.append(inputs[0])
        # for i in range(1, self.num_ins):
        #     outs.append(
        #         F.interpolate(inputs[i], scale_factor=2**i, mode='bilinear'))
        # out = torch.cat(outs, dim=1)
        # if out.requires_grad and self.with_cp:
        #     out = checkpoint(self.reduction_conv, out)
        # else:
        #     out = self.reduction_conv(out)
        # outs = [out]
        # for i in range(1, self.num_outs):
        #     outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))
        # outputs = []
        #
        # for i in range(self.num_outs):
        #     if outs[i].requires_grad and self.with_cp:
        #         tmp_out = checkpoint(self.fpn_convs[i], outs[i])
        #     else:
        #         tmp_out = self.fpn_convs[i](outs[i])
        #     outputs.append(tmp_out)
        # for i in outputs:
        #     print(i.size())
        return tuple(outputs)
