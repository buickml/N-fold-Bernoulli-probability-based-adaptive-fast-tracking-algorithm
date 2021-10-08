import torch as tr
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (tr.tanh(F.softplus(x)))


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride=(1, 1), p = None):
        super(Conv, self).__init__()
        self.C = nn.Conv2d(in_channel, out_channel, kernel, stride, autopad(kernel, p))

    def forward(self, x):
        return self.C(x)


class CBa(nn.Module):
    def __init__(self, act, in_channel, out_channel, kernel, stride=(1, 1), p = None, extra=None):
        super(CBa, self).__init__()
        self.kernel = kernel
        if kernel == (1, 1):
            self.C = nn.Conv2d(in_channel, out_channel, kernel, stride, autopad(kernel, p))
            self.B = nn.BatchNorm2d(out_channel)

        elif kernel == (3, 3):
            self.C33 = nn.Conv2d(in_channel, out_channel, kernel, stride, autopad(kernel, p))
            self.B33 = nn.BatchNorm2d(out_channel)
            self.C13 = nn.Conv2d(in_channel, out_channel, (1,3), stride, autopad(1, p))
            self.B13 = nn.BatchNorm2d(out_channel)
            self.C31 = nn.Conv2d(in_channel, out_channel, (3, 1), stride, autopad(1, p))
            self.B31 = nn.BatchNorm2d(out_channel)
        if extra:
            self.a = act(**extra)
        else:
            self.a = act()
        self.out_channel = out_channel

    def forward(self, x):
        if self.kernel == (1, 1):
            x = self.C(x)
            x = self.B(x)
            x = self.a(x)
        elif self.kernel == (3, 3):
            x33 = self.C33(x)
            x33 = self.B33(x33)
            x33 = self.a(x33)

            x13 = self.C13(x)
            x13 = self.B13(x13)
            x13 = self.a(x13)

            x31 = self.C31(x)
            x31 = self.B31(x31)
            x31 = self.a(x31)

            x = tr.cat((x33, x13, x31), dim=1)
        return x


class DensUnit(nn.Module):
    def __init__(self, act, in_channel, out_channels, kernels, p=None, extra=None):
        super(DensUnit, self).__init__()
        self.cbas = []
        for out_channel, k in zip(out_channels, kernels):
            self.cbas.append(CBa(act, in_channel, out_channel, k, extra=extra))
            in_channel = self.cbas[-1].out_channel
        self.CBas = nn.Sequential(*self.cbas)
        self.out_channel = out_channels[0] + out_channels[1]
            

    def forward(self, x):
        tmp_out = self.CBas(x)
        return tr.cat((x, tmp_out), dim=1)


class DensBlock(nn.Module):
    def __init__(self, act, bn, in_channel, out_channels, kernels, p=None, extra=None):
        super(DensBlock, self).__init__()
        self.blocks = [CBa(act, in_channel, out_channels[-1], kernels[0], extra=extra)]
        in_channel = self.blocks[-1].out_channel
        if len(kernels) == len(out_channels)+1:
            kernels = kernels[1:]
        elif len(out_channels) % len(kernels) != 0:
            kernels = kernels**(len(out_channels)//kernels)
        for i in range(bn):
            self.blocks.append(DensUnit(act, in_channel, out_channels, kernels, extra=extra))
            in_channel = self.blocks[-1].out_channel
        self.out_channel = self.blocks[-1].out_channel
        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x):
        return self.blocks(x)


class CSPx(nn.Module):
    def __init__(self, act, bn, in_channel, out_channels, kernels, strides=(2,1,1), p = None, extra=None):
        super(CSPx, self).__init__()
        res_channels = out_channels[1]
        res_kernels = kernels[1]
        self.f_cba = CBa(act, in_channel, out_channels[0], kernels[0], strides[0], p, extra)
        nin_channel = self.f_cba.out_channel
        self.bypass_cba = CBa(act, self.f_cba.out_channel, out_channels[-1], kernels[-1], strides[-1], p, extra)
        self.res_block = DensBlock(act, bn, nin_channel, res_channels, res_kernels, extra=None)
        #print(".......", out_channels[-2])
        self.l_cba = CBa(act, self.res_block.out_channel, out_channels[-2], kernels[-2], strides[-2], p, extra)
        self.out_channel = self.bypass_cba.out_channel + self.l_cba.out_channel

    def forward(self, x):
        x = self.f_cba(x)
        bypass_out = self.bypass_cba(x)
        x = self.res_block(x)
        return tr.cat((bypass_out, self.l_cba(x)), 1)


class CSPx0(nn.Module):
    def __init__(self, act, in_channel, out_channels, kernels, strides=1, p = None, extra=None):
        super(CSPx0, self).__init__()
        res_channels = out_channels[1:-2]
        res_kernels = kernels[1:-2]
        res_strides = strides[1:-2]
        self.f_cba = CBa(act, in_channel, out_channels[0], kernels[0], strides[0], p, extra)
        self.bypass_cba = CBa(act, in_channel, out_channels[-1], kernels[-1], strides[-1], p, extra)
        in_channel = out_channels[0]
        self.res_blocks = []   #000000000000000000000000000
        for channel, kernel, stride in zip(res_channels, res_kernels, res_strides):
            self.res_blocks.append(act, in_channel, channel, kernel, stride, p, extra)
            in_channel = channel
        self.res_blocks = nn.Sequential(*self.res_blocks)
        self.l_cba = CBa(act, in_channel, out_channels[-2], kernels[-2], strides[-2], p, extra)

    def forward(self, x):
        x = self.f_cba(x)
        bypass_out = self.bypass_cba(x)
        x = self.res_blocks(x)
        return tr.cat((bypass_out, self.l_cba(x)), 1)


class SPP(nn.Module):
    def __init__(self, kernels, p=None):
        super(SPP, self).__init__()
        self.max_pools = []
        for kernel in kernels:
            self.max_pools.append(nn.MaxPool2d(kernel, 1, autopad(kernel, p)))

    def forward(self, x):
        outputs = []
        for max_pool in self.max_pools:
            outputs.append(max_pool(x))
        for o in outputs:
            x = tr.cat((x, o), dim=1)
        return x
